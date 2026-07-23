#include "world/terrain_bootstrap.hpp"

#import <Foundation/Foundation.h>

#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <mutex>
#include <string>
#include <system_error>
#include <unistd.h>

@interface RycraftTerrainDownloadDelegate : NSObject <NSURLSessionDataDelegate> {
@public
    std::mutex mutex_;
    std::condition_variable condition_;
    worldgen::bootstrap::TerrainDownloadProgress progress_;
    const worldgen::bootstrap::TerrainBootstrapCancellation* cancellation_;
    NSURLSessionDataTask* task_;
    int descriptor_;
    uint64_t resumeOffset_;
    uint64_t writtenBytes_;
    uint64_t expectedBytes_;
    bool completed_;
    bool published_;
    bool canceled_;
    bool responseAccepted_;
    std::string errorMessage_;
}

- (instancetype)initWithDescriptor:(int)descriptor
                      resumeOffset:(uint64_t)resumeOffset
                     expectedBytes:(uint64_t)expectedBytes
                          progress:(worldgen::bootstrap::TerrainDownloadProgress)progress
                      cancellation:
                          (const worldgen::bootstrap::TerrainBootstrapCancellation*)cancellation;

@end

@implementation RycraftTerrainDownloadDelegate

- (instancetype)initWithDescriptor:(int)descriptor
                      resumeOffset:(uint64_t)resumeOffset
                     expectedBytes:(uint64_t)expectedBytes
                          progress:(worldgen::bootstrap::TerrainDownloadProgress)progress
                      cancellation:
                          (const worldgen::bootstrap::TerrainBootstrapCancellation*)cancellation {
    self = [super init];
    if (self != nil) {
        progress_ = std::move(progress);
        cancellation_ = cancellation;
        task_ = nil;
        descriptor_ = descriptor;
        resumeOffset_ = resumeOffset;
        writtenBytes_ = resumeOffset;
        expectedBytes_ = expectedBytes;
        completed_ = false;
        published_ = false;
        canceled_ = false;
        responseAccepted_ = false;
    }
    return self;
}

- (void)dealloc {
    if (descriptor_ >= 0) {
        ::close(descriptor_);
        descriptor_ = -1;
    }
#if !__has_feature(objc_arc)
    [super dealloc];
#endif
}

- (void)URLSession:(NSURLSession*)session
              dataTask:(NSURLSessionDataTask*)dataTask
    didReceiveResponse:(NSURLResponse*)response
     completionHandler:(void (^)(NSURLSessionResponseDisposition))completionHandler {
    (void)session;
    (void)dataTask;
    NSHTTPURLResponse* httpResponse = [response isKindOfClass:[NSHTTPURLResponse class]]
                                          ? static_cast<NSHTTPURLResponse*>(response)
                                          : nil;
    const NSInteger status = httpResponse == nil ? 0 : httpResponse.statusCode;
    bool accepted = status == 200 || status == 206;
    if (accepted && status == 206 && resumeOffset_ > 0) {
        NSString* contentRange = [httpResponse valueForHTTPHeaderField:@"Content-Range"];
        NSString* expectedPrefix = [NSString
            stringWithFormat:@"bytes %llu-", static_cast<unsigned long long>(resumeOffset_)];
        accepted = contentRange != nil && [contentRange hasPrefix:expectedPrefix];
        if (!accepted)
            errorMessage_ = "Terrain model server returned an incompatible byte range";
    } else if (accepted && status == 200 && resumeOffset_ > 0) {
        if (::ftruncate(descriptor_, 0) != 0 || ::lseek(descriptor_, 0, SEEK_SET) < 0) {
            accepted = false;
            errorMessage_ = "Could not restart a terrain download after the server ignored its "
                            "byte range";
        } else {
            resumeOffset_ = 0;
            writtenBytes_ = 0;
        }
    }
    if (!accepted && errorMessage_.empty()) {
        errorMessage_ = status == 0 ? "Terrain model server did not return an HTTP response"
                                    : "Terrain model server returned HTTP status " +
                                          std::to_string(static_cast<long long>(status));
    }
    responseAccepted_ = accepted;
    completionHandler(accepted ? NSURLSessionResponseAllow : NSURLSessionResponseCancel);
}

- (void)URLSession:(NSURLSession*)session
          dataTask:(NSURLSessionDataTask*)dataTask
    didReceiveData:(NSData*)data {
    (void)session;
    if (!responseAccepted_ || descriptor_ < 0 || data.length == 0)
        return;
    const auto* bytes = static_cast<const uint8_t*>(data.bytes);
    size_t remaining = static_cast<size_t>(data.length);
    while (remaining > 0) {
        const ssize_t written = ::write(descriptor_, bytes, remaining);
        if (written < 0 && errno == EINTR)
            continue;
        if (written <= 0) {
            errorMessage_ = "Could not write the staged terrain model download";
            [dataTask cancel];
            return;
        }
        bytes += written;
        remaining -= static_cast<size_t>(written);
        writtenBytes_ += static_cast<uint64_t>(written);
    }
    if (writtenBytes_ > expectedBytes_) {
        errorMessage_ = "Terrain model download exceeded the pinned asset size";
        [dataTask cancel];
        return;
    }
    if ((cancellation_ != nullptr && cancellation_->canceled()) ||
        (progress_ && !progress_(writtenBytes_))) {
        canceled_ = true;
        [dataTask cancel];
    }
}

- (void)URLSession:(NSURLSession*)session
                    task:(NSURLSessionTask*)task
    didCompleteWithError:(NSError*)error {
    (void)session;
    (void)task;
    {
        std::lock_guard lock(mutex_);
        const bool synchronized = descriptor_ >= 0 && ::fsync(descriptor_) == 0;
        const bool closed = descriptor_ >= 0 && ::close(descriptor_) == 0;
        descriptor_ = -1;
        if (error != nil) {
            const bool cancellationRequested =
                cancellation_ != nullptr && cancellation_->canceled();
            canceled_ = canceled_ || ([error code] == NSURLErrorCancelled && cancellationRequested);
            if (!canceled_ && errorMessage_.empty()) {
                const char* message = [[error localizedDescription] UTF8String];
                errorMessage_ = message != nullptr ? message : "Terrain model download failed";
            }
        }
        if (!synchronized || !closed) {
            if (errorMessage_.empty())
                errorMessage_ = "Could not finish the staged terrain model download";
        } else if (error == nil && errorMessage_.empty() && responseAccepted_ &&
                   writtenBytes_ == expectedBytes_) {
            published_ = true;
        } else if (error == nil && errorMessage_.empty()) {
            errorMessage_ = "Terrain model download ended before the pinned asset was complete";
        }
        completed_ = true;
    }
    condition_.notify_all();
}

@end

namespace worldgen::bootstrap {

namespace {

class AppleTerrainModelTransport final : public TerrainModelTransport {
public:
    TerrainTransferResult download(const TerrainAssetSpec& asset,
                                   const std::filesystem::path& destination,
                                   const TerrainDownloadProgress& progress,
                                   const TerrainBootstrapCancellation& cancellation) override {
        @autoreleasepool {
            if (cancellation.canceled())
                return TerrainTransferResult::cancellation();

            std::error_code filesystemError;
            std::filesystem::create_directories(destination.parent_path(), filesystemError);
            if (filesystemError) {
                return TerrainTransferResult::failure(
                    "Could not create the terrain download staging directory: " +
                    filesystemError.message());
            }

            uint64_t resumeOffset = 0;
            const bool stagedExists = std::filesystem::exists(destination, filesystemError);
            if (!filesystemError && stagedExists &&
                std::filesystem::is_regular_file(destination, filesystemError)) {
                const uintmax_t size = std::filesystem::file_size(destination, filesystemError);
                if (!filesystemError && size <= asset.byteSize)
                    resumeOffset = static_cast<uint64_t>(size);
            }
            if (filesystemError || resumeOffset > asset.byteSize) {
                return TerrainTransferResult::failure(
                    "Could not inspect the staged terrain model download");
            }
            const int descriptor =
                ::open(destination.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, S_IRUSR | S_IWUSR);
            if (descriptor < 0 ||
                ::lseek(descriptor, static_cast<off_t>(resumeOffset), SEEK_SET) < 0) {
                if (descriptor >= 0)
                    ::close(descriptor);
                return TerrainTransferResult::failure(
                    "Could not open the staged terrain model download");
            }

            NSString* urlString = [NSString stringWithUTF8String:asset.url.c_str()];
            NSURL* url = [NSURL URLWithString:urlString];
            if (url == nil) {
                ::close(descriptor);
                return TerrainTransferResult::failure("Pinned terrain asset URL is invalid");
            }

            RycraftTerrainDownloadDelegate* delegate =
                [[RycraftTerrainDownloadDelegate alloc] initWithDescriptor:descriptor
                                                              resumeOffset:resumeOffset
                                                             expectedBytes:asset.byteSize
                                                                  progress:progress
                                                              cancellation:&cancellation];
            if (delegate == nil) {
                ::close(descriptor);
                return TerrainTransferResult::failure(
                    "Could not prepare the terrain model download");
            }
            NSURLSessionConfiguration* configuration =
                [NSURLSessionConfiguration ephemeralSessionConfiguration];
            configuration.URLCache = nil;
            configuration.requestCachePolicy = NSURLRequestReloadIgnoringLocalCacheData;
            configuration.timeoutIntervalForRequest = 120.0;
            configuration.timeoutIntervalForResource = 7.0 * 24.0 * 60.0 * 60.0;

            NSOperationQueue* delegateQueue = [[NSOperationQueue alloc] init];
            delegateQueue.maxConcurrentOperationCount = 1;
            NSURLSession* session = [NSURLSession sessionWithConfiguration:configuration
                                                                  delegate:delegate
                                                             delegateQueue:delegateQueue];
#if !__has_feature(objc_arc)
            [session retain];
#endif
            NSMutableURLRequest* request = [NSMutableURLRequest requestWithURL:url];
            request.HTTPMethod = @"GET";
            [request setValue:@"identity" forHTTPHeaderField:@"Accept-Encoding"];
            if (resumeOffset > 0) {
                NSString* range = [NSString
                    stringWithFormat:@"bytes=%llu-", static_cast<unsigned long long>(resumeOffset)];
                [request setValue:range forHTTPHeaderField:@"Range"];
            }
            NSURLSessionDataTask* task = [session dataTaskWithRequest:request];
            delegate->task_ = task;
            [task resume];

            {
                std::unique_lock lock(delegate->mutex_);
                while (!delegate->completed_) {
                    delegate->condition_.wait_for(lock, std::chrono::milliseconds(100));
                    if (cancellation.canceled()) {
                        [task cancel];
                    }
                }
            }

            const bool canceled = delegate->canceled_ || cancellation.canceled();
            const bool succeeded = delegate->published_ && delegate->errorMessage_.empty();
            const std::string errorMessage = delegate->errorMessage_;
            [session finishTasksAndInvalidate];
#if !__has_feature(objc_arc)
            [session release];
            [delegateQueue release];
            [delegate release];
#endif

            if (canceled)
                return TerrainTransferResult::cancellation();
            if (!succeeded) {
                return TerrainTransferResult::failure(
                    errorMessage.empty() ? "Terrain model download failed" : errorMessage);
            }
            if (progress)
                progress(asset.byteSize);
            return TerrainTransferResult::success();
        }
    }
};

} // namespace

std::filesystem::path defaultRycraftApplicationSupportPath() {
    if (const char* overridePath = std::getenv("RYCRAFT_APPLICATION_SUPPORT_ROOT")) {
        const std::filesystem::path requested(overridePath);
        if (!requested.empty() && requested.is_absolute())
            return requested;
    }
    @autoreleasepool {
        NSArray<NSURL*>* locations =
            [[NSFileManager defaultManager] URLsForDirectory:NSApplicationSupportDirectory
                                                   inDomains:NSUserDomainMask];
        NSURL* location = [locations firstObject];
        if (location != nil) {
            NSString* path = [[location URLByAppendingPathComponent:@"rycraft"] path];
            if (path != nil)
                return std::filesystem::path([path UTF8String]);
        }
        NSString* fallback =
            [[NSHomeDirectory() stringByAppendingPathComponent:@"Library/Application Support"]
                stringByAppendingPathComponent:@"rycraft"];
        return std::filesystem::path([fallback UTF8String]);
    }
}

std::unique_ptr<TerrainModelTransport> makeAppleTerrainModelTransport() {
    return std::make_unique<AppleTerrainModelTransport>();
}

} // namespace worldgen::bootstrap
