#include <sasktran2/raytracing.h>
#include "sasktran2/source_algorithms.h"
#include <sasktran2/emission_source.h>
#include <sasktran2/dual.h>
#include <sasktran2/config.h>
#include <sasktran2/atmosphere/atmosphere.h>

namespace sasktran2::emission {
    template <int NSTOKES>
    void EmissionSource<NSTOKES>::initialize_geometry(
        const std::vector<sasktran2::raytracing::TracedRay>& los_rays) {
        // Store the rays for later
        m_los_rays = &los_rays;
    }

    template <int NSTOKES>
    void EmissionSource<NSTOKES>::initialize_atmosphere(
        const sasktran2::atmosphere::Atmosphere<NSTOKES>& atmosphere) {
        // Store the atmosphere for later
        m_atmosphere = &atmosphere;
    }

    template <int NSTOKES>
    void EmissionSource<NSTOKES>::integrated_source_constant(
        int wavelidx, int losidx, int layeridx, int wavel_threadidx,
        int threadidx, const sasktran2::raytracing::SphericalLayer& layer,
        const sasktran2::SparseODDualView& shell_od,
        sasktran2::Dual<double, sasktran2::dualstorage::dense, NSTOKES>& source)
        const {
        // Integrates assuming the source is constant in the layer and
        // determined by the average of the layer boundaries

        double ssa_start = 0;
        double ssa_end = 0;
        double emission_start = 0;
        double emission_end = 0;

        // Calculate SSA and emission at the layer boundaries
        for (auto& ele : layer.entrance.interpolation_weights) {
            ssa_start +=
                m_atmosphere->storage().ssa(ele.first, wavelidx) * ele.second;
            emission_start +=
                m_atmosphere->storage().emission_source(ele.first, wavelidx) *
                ele.second;
        }
        for (auto& ele : layer.exit.interpolation_weights) {
            ssa_end +=
                m_atmosphere->storage().ssa(ele.first, wavelidx) * ele.second;
            emission_end +=
                m_atmosphere->storage().emission_source(ele.first, wavelidx) *
                ele.second;
        }

        // Average value of layer boundaries
        double source_factor1 = (1 - shell_od.exp_minus_od);
        double emission_cell =
            source_factor1 *
            ((1 - ssa_start) * emission_start * layer.od_quad_start_fraction +
             (1 - ssa_end) * emission_end * layer.od_quad_end_fraction);

        if constexpr (NSTOKES == 1) {
            source.value.array() += emission_cell;
        } else {
            source.value(0) += emission_cell;
        }
    }

    template <int NSTOKES>
    void EmissionSource<NSTOKES>::integrated_source(
        int wavelidx, int losidx, int layeridx, int wavel_threadidx,
        int threadidx, const sasktran2::raytracing::SphericalLayer& layer,
        const sasktran2::SparseODDualView& shell_od,
        sasktran2::Dual<double, sasktran2::dualstorage::dense, NSTOKES>& source)
        const {
        if (layer.layer_distance < MINIMUM_SHELL_SIZE_M) {
            // Essentially an empty shell from rounding, don't have to do
            // anything
            return;
        }

        integrated_source_constant(wavelidx, losidx, layeridx, wavel_threadidx,
                                   threadidx, layer, shell_od, source);
    }

    template <int NSTOKES>
    void EmissionSource<NSTOKES>::end_of_ray_source(
        int wavelidx, int losidx, int wavel_threadidx, int threadidx,
        sasktran2::Dual<double, sasktran2::dualstorage::dense, NSTOKES>& source)
        const {
        if (m_los_rays->at(losidx).ground_is_hit) {
            double emission_surface =
                m_atmosphere->surface().emission()[wavelidx];
            if constexpr (NSTOKES == 1) {
                source.value.array() += emission_surface;
            } else {
                source.value(0) += emission_surface;
            }
        }
    }

    template class EmissionSource<1>;
    template class EmissionSource<3>;

} // namespace sasktran2::emission