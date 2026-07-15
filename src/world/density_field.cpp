#include "world/density_field.hpp"

#include "world/gen_seeds.hpp"
#include "world/macro_generation.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

DensityField::DensityField(uint32_t worldSeed)
    : detail_(genseed::subSeed(worldSeed, genseed::DETAIL_3D))
    , cheese_(genseed::subSeed(worldSeed, genseed::CHEESE))
    , spaghetti1_(genseed::subSeed(worldSeed, genseed::SPAGHETTI_1))
    , spaghetti2_(genseed::subSeed(worldSeed, genseed::SPAGHETTI_2))
    , noodle1_(genseed::subSeed(worldSeed, genseed::NOODLE_1))
    , noodle2_(genseed::subSeed(worldSeed, genseed::NOODLE_2)) {}

DensityColumnContext DensityField::columnContext(double x, double z,
                                                 const worldgen::GeologySample& geology) const {
    DensityColumnContext result;
    if (geology.rock == worldgen::RockType::LIMESTONE) {
        const double region = cheese_.octave2D(x / 310.0, z / 310.0, 2);
        result.karstRegionStrength = std::clamp((region + 0.08) / 0.48, 0.0, 1.0);
    }
    if (geology.boundary == worldgen::PlateBoundary::TRANSFORM) {
        result.faultStrength = std::clamp(geology.faultStrength, 0.0, 1.0);
    }
    return result;
}

double DensityField::density(double x, double y, double z, const ColumnShape& col,
                             const DensityColumnContext& context) const {
    // ---- Terrain: signed distance below the column surface, distorted by
    // anisotropic 3D detail (wider than tall → ledges and overhangs).
    double d = col.height - y;
    if (col.detailAmp > 0.0) {
        d += detail_.octave3D(x / 70.0, y / 48.0, z / 70.0, 2) * col.detailAmp;
    }
    d = std::clamp(d, -DENSITY_CAP, DENSITY_CAP);

    // More than a cell above the terrain, caves cannot change the sign —
    // skip their noise entirely (roughly 40% of lattice points).
    if (d <= -4.0) return d;

    // ---- Near-surface sealing: cave strength s ramps 0 → 1 over the 12
    // blocks below the surface, so tunnels don't riddle the topsoil…
    double s;
    bool waterCovered = col.height < 66.0 || col.ravineFloor < 64.0;
    if (waterCovered) {
        // …and under water (oceans, rivers, flooded ravines) at least 8
        // blocks of cover always remain: no drained seas.
        double sealRef = col.height - 46.0 * col.ravineEdge;
        s = std::clamp((sealRef - y - 8.0) / 12.0, 0.0, 1.0);
    } else if (col.entrance > 0.4 && col.height >= 70.0) {
        // …except where the entrance mask keeps full strength up to the
        // surface: natural cave mouths on high-and-dry terrain only.
        s = 1.0;
    } else {
        s = std::clamp((col.height - y) / 12.0, 0.0, 1.0);
    }

    double depthFrac = std::clamp((64.0 - y) / 64.0, 0.0, 1.0);

    // ---- Cheese caverns: carve where the fBm exceeds a threshold that
    // drops with depth (bigger rooms down deep) and rises near the surface.
    double cheeseThreshold = 0.42 - 0.16 * depthFrac + (1.0 - s) * 0.6;
    double cheese = cheese_.octave3D(x / 90.0, y / 60.0, z / 90.0, 2);
    double dCheese = std::clamp((cheeseThreshold - cheese) * 40.0, -DENSITY_CAP, DENSITY_CAP);

    // ---- Spaghetti tunnels: the intersection band where two independent
    // noises are BOTH near zero traces long winding tubes.
    double spagWidth = (0.055 + 0.025 * depthFrac) * s;
    double sp1 = std::abs(spaghetti1_.noise3D(x / 68.0, y / 44.0, z / 68.0));
    double sp2 = std::abs(spaghetti2_.noise3D(x / 68.0, y / 44.0, z / 68.0));
    double dSpaghetti =
        std::clamp((std::max(sp1, sp2) - spagWidth) * 150.0, -DENSITY_CAP, DENSITY_CAP);

    // ---- Noodle crawls: same construction, tighter and only deep down.
    double dNoodle = DENSITY_CAP;
    if (y < 64.0) {
        double noodleWidth = 0.04 * s;
        double n1 = std::abs(noodle1_.noise3D(x / 34.0, y / 22.0, z / 34.0));
        double n2 = std::abs(noodle2_.noise3D(x / 34.0, y / 22.0, z / 34.0));
        dNoodle = std::clamp((std::max(n1, n2) - noodleWidth) * 200.0, -DENSITY_CAP, DENSITY_CAP);
    }

    // ---- Ravines: per-column canyon cut. The (1 - edge) offset lifts the
    // cut out of reach at the canyon lip so shallow edges taper to gullies
    // instead of outlining every ravine with a one-block trench.
    double dRavine = DENSITY_CAP;
    if (col.ravineEdge > 0.001) {
        dRavine = std::clamp((col.ravineFloor - y) * 3.0 + (1.0 - col.ravineEdge) * 64.0,
                             -DENSITY_CAP, DENSITY_CAP);
    }

    // ---- Limestone karst: broad solution cavities follow bedding planes,
    // while slowly varying spaghetti fields form sink shafts. A regional
    // gate keeps limestone from becoming uniformly hollow. Surface sealing
    // still applies, including the stronger seal below generated water.
    double dKarst = DENSITY_CAP;
    if (context.karstRegionStrength > 0.0) {
        const double karstSurfaceSeal = std::clamp((col.height - y - 6.0) / 10.0, 0.0, 1.0);
        const double regionStrength = context.karstRegionStrength * std::min(s, karstSurfaceSeal);
        if (regionStrength > 0.001) {
            const double chamberA = std::abs(cheese_.noise3D(x / 78.0, y / 46.0, z / 78.0));
            const double chamberB = std::abs(spaghetti1_.noise3D(x / 112.0, y / 68.0, z / 112.0));
            const double chamberWidth = (0.075 + depthFrac * 0.045) * regionStrength;
            const double chamber = (std::max(chamberA, chamberB * 0.74) - chamberWidth) * 185.0;

            const double foldedY = y + cheese_.noise2D(x / 150.0, z / 150.0) * 7.0;
            const double beddingWave = std::abs(std::sin(foldedY * std::numbers::pi / 13.0));
            const double beddingBreak =
                std::abs(spaghetti2_.noise3D(x / 96.0, y / 210.0, z / 96.0));
            const double beddingWidth = (0.050 + depthFrac * 0.025) * regionStrength;
            const double bedding =
                (std::max(beddingWave, beddingBreak * 0.58) - beddingWidth) * 115.0;

            const double shaftA = std::abs(spaghetti1_.noise3D(x / 58.0, y / 820.0, z / 58.0));
            const double shaftB = std::abs(noodle1_.noise3D(x / 91.0, y / 690.0, z / 91.0));
            const double shaftWidth = (0.030 + depthFrac * 0.024) * regionStrength;
            const double shaft = (std::max(shaftA, shaftB) - shaftWidth) * 210.0;
            dKarst = std::clamp(std::min({chamber, bedding, shaft}), -DENSITY_CAP, DENSITY_CAP);
        }
    }

    // ---- Transform faults: a narrow, vertically coherent zero-isosurface
    // produces deep fault caves. On dry ground the fissure can reach the
    // surface as a ravine, while the ordinary water seal protects oceans and
    // lakes from being drained by the fault field.
    double dFault = DENSITY_CAP;
    if (context.faultStrength > 0.001) {
        const double seal = waterCovered ? s : 1.0;
        const double nearSurface = std::clamp((y - (col.height - 34.0)) / 30.0, 0.0, 1.0);
        const double width = context.faultStrength *
                             (0.010 + context.faultStrength * 0.052 + nearSurface * 0.012) * seal;
        const double trace = std::abs(spaghetti2_.noise3D(x / 168.0, y / 880.0, z / 168.0));
        const double fracture = std::abs(noodle2_.noise3D(x / 74.0, y / 510.0, z / 74.0));
        const double sheet = std::min(trace, fracture * 0.82);
        dFault = std::clamp((sheet - width) * 190.0, -DENSITY_CAP, DENSITY_CAP);
    }

    return std::min({d, dCheese, dSpaghetti, dNoodle, dRavine, dKarst, dFault});
}

bool DensityField::supportsCaveEcotope(double x, double z, double terrainHeight,
                                       const worldgen::GeologySample& geology) const {
    const double probeY = terrainHeight - 26.0;
    if (geology.rock == worldgen::RockType::LIMESTONE) {
        const double region = cheese_.octave2D(x / 310.0, z / 310.0, 2);
        if (region > -0.08) {
            const double chamberA = std::abs(cheese_.noise3D(x / 78.0, probeY / 46.0, z / 78.0));
            const double chamberB =
                std::abs(spaghetti1_.noise3D(x / 112.0, probeY / 68.0, z / 112.0));
            const double shaftA = std::abs(spaghetti1_.noise3D(x / 58.0, probeY / 820.0, z / 58.0));
            const double shaftB = std::abs(noodle1_.noise3D(x / 91.0, probeY / 690.0, z / 91.0));
            if (std::max(chamberA, chamberB * 0.74) < 0.16 || std::max(shaftA, shaftB) < 0.075) {
                return true;
            }
        }
    }

    if (geology.boundary == worldgen::PlateBoundary::TRANSFORM && geology.faultStrength > 0.08) {
        const double trace = std::abs(spaghetti2_.noise3D(x / 168.0, probeY / 880.0, z / 168.0));
        const double fracture = std::abs(noodle2_.noise3D(x / 74.0, probeY / 510.0, z / 74.0));
        return std::min(trace, fracture * 0.82) < 0.016 + geology.faultStrength * 0.052;
    }
    return false;
}
