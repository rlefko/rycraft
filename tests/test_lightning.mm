#include <audio/sfx.hpp>
#include <audio/thunder.hpp>
#include <render/lightning_renderer.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>

TEST_CASE("Lightning flash timing is deterministic and bounded", "[lightning][render]") {
    LightningEvent event;
    event.tick = 100;
    event.intensity = 0.8F;

    REQUIRE(lightningFlashIntensity(event, 99) == 0.0F);
    REQUIRE(lightningFlashIntensity(event, 100) == Catch::Approx(0.8F).margin(0.001F));
    REQUIRE(lightningFlashIntensity(event, 102) > 0.0F);
    REQUIRE(lightningFlashIntensity(event, 112) == 0.0F);
    REQUIRE(lightningFlashIntensity(event, 102) == lightningFlashIntensity(event, 102));
    REQUIRE(lightningFlashIntensity(event, 100, 0.0F) == 0.0F);
}

TEST_CASE("Lightning topology derives bounded branches from the event ID",
          "[lightning][render][determinism]") {
    REQUIRE(lightningBoltSegmentCount(0) == 64);
    REQUIRE(lightningBoltSegmentCount(1) == 72);
    REQUIRE(lightningBoltSegmentCount(2) == 80);
    for (uint64_t eventId = 0; eventId < 256; ++eventId) {
        const uint32_t segments = lightningBoltSegmentCount(eventId);
        REQUIRE(segments >= 64);
        REQUIRE(segments <= 80);
        REQUIRE((segments - 48) % 8 == 0);
    }
}

TEST_CASE("Procedural thunder is deterministic finite and event-specific",
          "[audio][thunder][determinism]") {
    const std::vector<float> first = SoundEffect::generateThunder(0x123456789ABCDEF0ULL, 1.0F);
    const std::vector<float> repeated = SoundEffect::generateThunder(0x123456789ABCDEF0ULL, 1.0F);
    const std::vector<float> different = SoundEffect::generateThunder(0x123456789ABCDEF1ULL, 1.0F);
    REQUIRE(first.size() == 4 * SoundEffect::SAMPLE_RATE);
    REQUIRE(first == repeated);
    REQUIRE(first != different);

    float peak = 0.0F;
    for (float sample : first) {
        peak = std::max(peak, std::abs(sample));
    }
    REQUIRE(std::all_of(first.begin(), first.end(), [](float sample) {
        return std::isfinite(sample) && sample >= -1.0F && sample <= 1.0F;
    }));
    REQUIRE(peak > 0.1F);

    const std::vector<float> silent = SoundEffect::generateThunder(42, 0.0F);
    REQUIRE(std::all_of(silent.begin(), silent.end(), [](float sample) { return sample == 0.0F; }));
}

TEST_CASE("Thunder scheduling uses sound travel time and suppresses replays",
          "[audio][thunder][scheduler]") {
    ThunderScheduler scheduler;
    scheduler.beginTimeline(100);

    LightningEvent oldEvent;
    oldEvent.id = 1;
    oldEvent.tick = 100;
    oldEvent.x = 343.0;
    oldEvent.intensity = 1.0F;
    REQUIRE_FALSE(scheduler.schedule(oldEvent, 0.0, 0.0, 0.0, 5.0));

    LightningEvent liveEvent = oldEvent;
    liveEvent.id = 2;
    liveEvent.tick = 101;
    REQUIRE(scheduler.schedule(liveEvent, 0.0, 0.0, 0.0, 5.0));
    REQUIRE_FALSE(scheduler.schedule(liveEvent, 0.0, 0.0, 0.0, 5.0));
    REQUIRE(scheduler.pendingCount() == 1);
    REQUIRE(scheduler.popDue(5.999).empty());

    const std::vector<ScheduledThunder> due = scheduler.popDue(6.0);
    REQUIRE(due.size() == 1);
    REQUIRE(due.front().eventId == liveEvent.id);
    REQUIRE(due.front().dueTimeSeconds == Catch::Approx(6.0));
    REQUIRE(due.front().distanceBlocks == Catch::Approx(343.0F));
    REQUIRE(due.front().gain > 0.0F);
    REQUIRE(scheduler.pendingCount() == 0);
    REQUIRE_FALSE(scheduler.schedule(liveEvent, 0.0, 0.0, 0.0, 7.0));
}

TEST_CASE("Generator v4 thunder scheduling converts block distance to physical meters",
          "[audio][thunder][scheduler][v4]") {
    ThunderScheduler scheduler;
    scheduler.beginTimeline(0, GENERATOR_V4_PHYSICAL_SCALE);
    LightningEvent event;
    event.id = 44;
    event.tick = 1;
    event.x = 343.0 / GENERATOR_V4_PHYSICAL_SCALE.horizontalMetersPerBlock;
    event.y = static_cast<float>(GENERATOR_V4_PHYSICAL_SCALE.altitudeDatumY);
    event.intensity = 1.0F;
    REQUIRE(scheduler.schedule(event, 0.0, GENERATOR_V4_PHYSICAL_SCALE.altitudeDatumY, 0.0, 2.0));
    REQUIRE(scheduler.popDue(2.999).empty());
    const std::vector<ScheduledThunder> due = scheduler.popDue(3.0);
    REQUIRE(due.size() == 1);
    REQUIRE(due.front().distanceMeters == Catch::Approx(343.0F));
    REQUIRE(due.front().distanceBlocks ==
            Catch::Approx(
                343.0F / static_cast<float>(GENERATOR_V4_PHYSICAL_SCALE.horizontalMetersPerBlock)));
}

TEST_CASE("Thunder scheduling remains bounded during a dense storm",
          "[audio][thunder][scheduler][bounds]") {
    ThunderScheduler scheduler;
    scheduler.beginTimeline(0);
    for (uint64_t index = 0; index < ThunderScheduler::MAX_PENDING; ++index) {
        LightningEvent event;
        event.id = index + 1;
        event.tick = index + 1;
        event.intensity = 1.0F;
        REQUIRE(scheduler.schedule(event, 0.0, 0.0, 0.0, 0.0));
    }
    LightningEvent overflow;
    overflow.id = 999;
    overflow.tick = 999;
    overflow.x = 34'300.0;
    overflow.intensity = 1.0F;
    REQUIRE_FALSE(scheduler.schedule(overflow, 0.0, 0.0, 0.0, 0.0));
    REQUIRE(scheduler.pendingCount() == ThunderScheduler::MAX_PENDING);

    // A dropped strike is still consumed. Re-querying it after the queue
    // drains must not replay an old deterministic weather bucket.
    REQUIRE(scheduler.popDue(1.0).size() == ThunderScheduler::MAX_PENDING);
    REQUIRE_FALSE(scheduler.schedule(overflow, 0.0, 0.0, 0.0, 2.0));
}

TEST_CASE("Thunder queue replaces a distant late strike with a nearer audible strike",
          "[audio][thunder][scheduler][priority]") {
    ThunderScheduler scheduler;
    scheduler.beginTimeline(0);
    for (uint64_t index = 0; index < ThunderScheduler::MAX_PENDING; ++index) {
        LightningEvent event;
        event.id = index + 1;
        event.tick = index + 1;
        event.x = 3'430.0 + static_cast<double>(index) * 343.0;
        event.intensity = 1.0F;
        REQUIRE(scheduler.schedule(event, 0.0, 0.0, 0.0, 0.0));
    }

    LightningEvent nearby;
    nearby.id = 1'000;
    nearby.tick = 1'000;
    nearby.x = 343.0;
    nearby.intensity = 1.0F;
    REQUIRE(scheduler.schedule(nearby, 0.0, 0.0, 0.0, 0.0));
    REQUIRE(scheduler.pendingCount() == ThunderScheduler::MAX_PENDING);

    const std::vector<ScheduledThunder> firstDue = scheduler.popDue(1.0);
    REQUIRE(firstDue.size() == 1);
    REQUIRE(firstDue.front().eventId == nearby.id);
    REQUIRE(firstDue.front().eventTick == nearby.tick);

    // The farthest strike was evicted, but its remembered ID prevents it from
    // returning later as delayed backlog after capacity becomes available.
    LightningEvent evicted;
    evicted.id = ThunderScheduler::MAX_PENDING;
    evicted.tick = ThunderScheduler::MAX_PENDING;
    evicted.x = 3'430.0 + static_cast<double>(ThunderScheduler::MAX_PENDING - 1) * 343.0;
    evicted.intensity = 1.0F;
    REQUIRE_FALSE(scheduler.schedule(evicted, 0.0, 0.0, 0.0, 2.0));
}

TEST_CASE("Thunder priority favors newer strikes when audible time and distance tie",
          "[audio][thunder][scheduler][priority]") {
    ThunderScheduler scheduler;
    scheduler.beginTimeline(0);
    for (uint64_t index = 0; index < ThunderScheduler::MAX_PENDING; ++index) {
        LightningEvent event;
        event.id = index + 1;
        event.tick = index + 1;
        event.x = 343.0;
        event.intensity = 1.0F;
        REQUIRE(scheduler.schedule(event, 0.0, 0.0, 0.0, 0.0));
    }

    LightningEvent newer;
    newer.id = 2'000;
    newer.tick = 2'000;
    newer.x = 343.0;
    newer.intensity = 1.0F;
    REQUIRE(scheduler.schedule(newer, 0.0, 0.0, 0.0, 0.0));

    const std::vector<ScheduledThunder> due = scheduler.popDue(1.0);
    REQUIRE(due.size() == ThunderScheduler::MAX_PENDING);
    REQUIRE(std::find_if(due.begin(), due.end(), [&](const ScheduledThunder& thunder) {
                return thunder.eventId == newer.id;
            }) != due.end());
    REQUIRE(std::find_if(due.begin(), due.end(), [](const ScheduledThunder& thunder) {
                return thunder.eventId == 1;
            }) == due.end());
}
