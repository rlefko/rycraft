#include "world/density_field.hpp"

#include "world/gen_seeds.hpp"

#include <algorithm>
#include <cmath>

DensityField::DensityField(uint32_t worldSeed)
    : detail_(genseed::subSeed(worldSeed, genseed::DETAIL_3D))
    , cheese_(genseed::subSeed(worldSeed, genseed::CHEESE))
    , spaghetti1_(genseed::subSeed(worldSeed, genseed::SPAGHETTI_1))
    , spaghetti2_(genseed::subSeed(worldSeed, genseed::SPAGHETTI_2))
    , noodle1_(genseed::subSeed(worldSeed, genseed::NOODLE_1))
    , noodle2_(genseed::subSeed(worldSeed, genseed::NOODLE_2)) {}

double DensityField::density(double x, double y, double z, const ColumnShape& col) const {
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

    return std::min({d, dCheese, dSpaghetti, dNoodle, dRavine});
}
