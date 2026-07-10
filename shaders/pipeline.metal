#include <metal_stdlib>
using namespace metal;

struct VertexIn {
  float4 position [[attribute(0)]];
};

struct VertexOut {
  float4 position [[position]];
  float4 color;
};

vertex VertexOut vertex_main(uint vertexID [[vertex_id]],
                              constant VertexIn* vertices [[buffer(1)]]) {
  VertexOut out;
  out.position = vertices[vertexID].position;
  out.color = float4(1.0, 1.0, 1.0, 1.0);
  return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
  return in.color;
}
