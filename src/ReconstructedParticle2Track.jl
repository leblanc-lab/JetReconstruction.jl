module ReconstructedParticle2Track

using JetReconstruction
using LorentzVectorHEP
using EDM4hep
using StaticArrays
using LinearAlgebra

"""
    getRP2TRK_mom(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the momentum magnitude of the track associated with each reconstructed particle.
"""
function getRP2TRK_mom(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, get_trackMom(tracks[p.tracks_begin+1]))
        else
            push!(result, NaN32)
        end
    end
    return result
end

"""
    getRP2TRK_charge(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the charge of each reconstructed particle that has an associated track.
"""
function getRP2TRK_charge(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, p.charge)
        else
            push!(result, NaN32)
        end
    end
    return result
end

"""
    getRP2TRK_Bz(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Calculate the magnetic field Bz for each reconstructed particle with an associated track.
"""
function getRP2TRK_Bz(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    c_light = 2.99792458e8  # speed of light in m/s
    a = c_light * 1e3 * 1e-15  # conversion factor for omega [1/mm]
    result = Float32[]
    
    for p in rps
        if p.tracks_begin < length(tracks)
            pt = sqrt(p.momentum.x^2 + p.momentum.y^2)
            sign_charge = p.charge > 0 ? 1.0 : -1.0
            Bz = tracks[p.tracks_begin+1].omega / a * pt * sign_charge
            push!(result, Bz)
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    Bz(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Float32

Calculate the magnetic field Bz from the first reconstructed particle with an associated track.
"""
function Bz(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    c_light = 2.99792458e8  # speed of light in m/s
    a = c_light * 1e3 * 1e-15  # conversion factor for omega [1/mm]
    Bz_value = -9.0f0
    
    for p in rps
        if p.tracks_begin < length(tracks)
            pt = sqrt(p.momentum.x^2 + p.momentum.y^2)
            sign_charge = p.charge > 0 ? 1.0 : -1.0
            Bz_value = tracks[p.tracks_begin+1].omega / a * pt * sign_charge
            break
        end
    end
    return Bz_value
end

"""
    XPtoPar_dxy(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}, V::TLorentzVector, Bz::Float32) -> Vector{Float32}

Calculate the transverse impact parameter dxy for each reconstructed particle relative to vertex V.
"""
function XPtoPar_dxy(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}, V::TLorentzVector, Bz::Float32)
    cSpeed = 2.99792458e8 * 1.0e-9  # speed of light in m/ns
    result = Float32[]
    
    for p in rps
        if p.tracks_begin < length(tracks)
            track = tracks[p.tracks_begin+1]
            D0_wrt0 = track.D0
            phi0_wrt0 = track.phi
            
            # Position vector at closest approach to (0,0,0)
            X = SVector{3}(-D0_wrt0 * sin(phi0_wrt0), D0_wrt0 * cos(phi0_wrt0), track.Z0)
            
            # Position vector relative to vertex V
            x = X - SVector{3}(V.x, V.y, V.z)
            
            # Momentum vector
            p_vec = SVector{3}(p.momentum.x, p.momentum.y, p.momentum.z)
            
            # Calculate impact parameter
            a = -p.charge * Bz * cSpeed
            pt = sqrt(p.momentum.x^2 + p.momentum.y^2)
            r2 = x[1]^2 + x[2]^2
            cross = x[1] * p_vec[2] - x[2] * p_vec[1]
            
            D = -9.0f0
            if pt^2 - 2 * a * cross + a^2 * r2 > 0
                T = sqrt(pt^2 - 2 * a * cross + a^2 * r2)
                if pt < 10.0
                    D = (T - pt) / a
                else
                    D = (-2 * cross + a * r2) / (T + pt)
                end
            end
            
            push!(result, D)
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    XPtoPar_dz(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}, V::TLorentzVector, Bz::Float32) -> Vector{Float32}

Calculate the longitudinal impact parameter dz for each reconstructed particle relative to vertex V.
"""
function XPtoPar_dz(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}, V::TLorentzVector, Bz::Float32)
    cSpeed = 2.99792458e8 * 1.0e-9  # speed of light in m/ns
    result = Float32[]
    
    for p in rps
        if p.tracks_begin < length(tracks)
            track = tracks[p.tracks_begin+1]
            D0_wrt0 = track.D0
            Z0_wrt0 = track.Z0
            phi0_wrt0 = track.phi
            
            # Position vector at closest approach to (0,0,0)
            X = SVector{3}(-D0_wrt0 * sin(phi0_wrt0), D0_wrt0 * cos(phi0_wrt0), Z0_wrt0)
            
            # Position vector relative to vertex V
            x = X - SVector{3}(V.x, V.y, V.z)
            
            # Momentum vector
            p_vec = SVector{3}(p.momentum.x, p.momentum.y, p.momentum.z)
            
            # Calculate dz parameter
            a = -p.charge * Bz * cSpeed
            pt = sqrt(p.momentum.x^2 + p.momentum.y^2)
            C = a / (2 * pt)
            r2 = x[1]^2 + x[2]^2
            cross = x[1] * p_vec[2] - x[2] * p_vec[1]
            T = sqrt(pt^2 - 2 * a * cross + a^2 * r2)
            
            D = 0.0f0
            if pt < 10.0
                D = (T - pt) / a
            else
                D = (-2 * cross + a * r2) / (T + pt)
            end
            
            B = C * sqrt(max(r2 - D^2, 0.0) / (1 + 2 * C * D))
            if abs(B) > 1.0
                B = sign(B)
            end
            
            st = asin(B) / C
            ct = p_vec[3] / pt
            
            dot_prod = x[1] * p_vec[1] + x[2] * p_vec[2]
            z0 = dot_prod > 0.0 ? x[3] - ct * st : x[3] + ct * st
            
            push!(result, z0)
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    XPtoPar_phi(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}, V::TLorentzVector, Bz::Float32) -> Vector{Float32}

Calculate the phi angle at the point of closest approach for each reconstructed particle relative to vertex V.
"""
function XPtoPar_phi(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}, V::TLorentzVector, Bz::Float32)
    cSpeed = 2.99792458e8 * 1.0e-9  # speed of light in m/ns
    result = Float32[]
    
    for p in rps
        if p.tracks_begin < length(tracks)
            track = tracks[p.tracks_begin+1]
            D0_wrt0 = track.D0
            Z0_wrt0 = track.Z0
            phi0_wrt0 = track.phi
            
            # Position vector at closest approach to (0,0,0)
            X = SVector{3}(-D0_wrt0 * sin(phi0_wrt0), D0_wrt0 * cos(phi0_wrt0), Z0_wrt0)
            
            # Position vector relative to vertex V
            x = X - SVector{3}(V.x, V.y, V.z)
            
            # Momentum vector
            p_vec = SVector{3}(p.momentum.x, p.momentum.y, p.momentum.z)
            
            # Calculate phi parameter
            a = -p.charge * Bz * cSpeed
            pt = sqrt(p.momentum.x^2 + p.momentum.y^2)
            r2 = x[1]^2 + x[2]^2
            cross = x[1] * p_vec[2] - x[2] * p_vec[1]
            T = sqrt(pt^2 - 2 * a * cross + a^2 * r2)
            
            phi0 = atan((p_vec[2] - a * x[1]) / T, (p_vec[1] + a * x[2]) / T)
            
            push!(result, phi0)
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    XPtoPar_C(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}, Bz::Float32) -> Vector{Float32}

Calculate the curvature parameter C for each reconstructed particle.
"""
function XPtoPar_C(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}, Bz::Float32)
    cSpeed = 2.99792458e8 * 1.0e3 * 1.0e-15  # conversion factor
    result = Float32[]
    
    for p in rps
        if p.tracks_begin < length(tracks)
            # Momentum vector
            p_vec = SVector{3}(p.momentum.x, p.momentum.y, p.momentum.z)
            
            sign_charge = p.charge > 0 ? 1.0 : -1.0
            a = sign_charge * Bz * cSpeed
            pt = sqrt(p.momentum.x^2 + p.momentum.y^2)
            C = a / (2 * pt)
            
            push!(result, C)
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    XPtoPar_ct(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}, Bz::Float32) -> Vector{Float32}

Calculate the cot(theta) parameter for each reconstructed particle.
"""
function XPtoPar_ct(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}, Bz::Float32)
    result = Float32[]
    
    for p in rps
        if p.tracks_begin < length(tracks)
            # Momentum vector
            p_vec = SVector{3}(p.momentum.x, p.momentum.y, p.momentum.z)
            pt = sqrt(p.momentum.x^2 + p.momentum.y^2)
            
            ct = p_vec[3] / pt
            
            push!(result, ct)
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK_D0(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the D0 parameter from the track associated with each reconstructed particle.
"""
function getRP2TRK_D0(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1].D0)
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK_D0_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the D0 covariance from the track associated with each reconstructed particle.
"""
function getRP2TRK_D0_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1].covMatrix[1])
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK_D0_sig(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the D0 significance from the track associated with each reconstructed particle.
"""
function getRP2TRK_D0_sig(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1].D0 / sqrt(tracks[p.tracks_begin+1].covMatrix[1]))
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK_Z0(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the Z0 parameter from the track associated with each reconstructed particle.
"""
function getRP2TRK_Z0(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1].Z0)
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK_Z0_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the Z0 covariance from the track associated with each reconstructed particle.
"""
function getRP2TRK_Z0_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1].covMatrix[10])
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK_phi(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the phi parameter from the track associated with each reconstructed particle.
"""
function getRP2TRK_phi(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1].phi)
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK_phi_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the phi covariance from the track associated with each reconstructed particle.
"""
function getRP2TRK_phi_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1].covMatrix[3])
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK_omega(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the omega parameter from the track associated with each reconstructed particle.
"""
function getRP2TRK_omega(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1].omega)
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK_omega_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the omega covariance from the track associated with each reconstructed particle.
"""
function getRP2TRK_omega_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1].covMatrix[6])
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK_tanLambda(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the tanLambda parameter from the track associated with each reconstructed particle.
"""
function getRP2TRK_tanLambda(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1].tanLambda)
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK_tanLambda_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the tanLambda covariance from the track associated with each reconstructed particle.
"""
function getRP2TRK_tanLambda_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1].covMatrix[15])
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK_d0_phi0_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the d0-phi0 covariance from the track associated with each reconstructed particle.
"""
function getRP2TRK_d0_phi0_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1].covMatrix[2])
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK_d0_omega_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the d0-omega covariance from the track associated with each reconstructed particle.
"""
function getRP2TRK_d0_omega_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1].covMatrix[4])
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK_d0_z0_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the d0-z0 covariance from the track associated with each reconstructed particle.
"""
function getRP2TRK_d0_z0_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1].covMatrix[7])
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK_d0_tanlambda_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the d0-tanlambda covariance from the track associated with each reconstructed particle.
"""
function getRP2TRK_d0_tanlambda_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1].covMatrix[11])
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK_phi0_omega_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the phi0-omega covariance from the track associated with each reconstructed particle.
"""
function getRP2TRK_phi0_omega_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1].covMatrix[5])
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK_phi0_z0_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the phi0-z0 covariance from the track associated with each reconstructed particle.
"""
function getRP2TRK_phi0_z0_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1].covMatrix[8])
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK_phi0_tanlambda_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the phi0-tanlambda covariance from the track associated with each reconstructed particle.
"""
function getRP2TRK_phi0_tanlambda_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1].covMatrix[12])
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK_omega_z0_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the omega-z0 covariance from the track associated with each reconstructed particle.
"""
function getRP2TRK_omega_z0_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1].covMatrix[9])
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK_omega_tanlambda_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the omega-tanlambda covariance from the track associated with each reconstructed particle.
"""
function getRP2TRK_omega_tanlambda_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1].covMatrix[13])
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK_z0_tanlambda_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Float32}

Get the z0-tanlambda covariance from the track associated with each reconstructed particle.
"""
function getRP2TRK_z0_tanlambda_cov(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Float32[]
    for p in rps
        if p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1].covMatrix[14])
        else
            push!(result, -9.0f0)
        end
    end
    return result
end

"""
    getRP2TRK(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{EDM4hep.TrackState}

Get the track states associated with each reconstructed particle.
"""
function getRP2TRK(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = EDM4hep.TrackState[]
    for p in rps
        if p.tracks_begin >= 0 && p.tracks_begin < length(tracks)
            push!(result, tracks[p.tracks_begin+1])
        end
    end
    return result
end

"""
    get_recoindTRK(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState}) -> Vector{Int}

Get the indices of reconstructed particles that have associated tracks.
"""
function get_recoindTRK(rps::Vector{EDM4hep.ReconstructedParticle}, tracks::Vector{EDM4hep.TrackState})
    result = Int[]
    for (i, p) in enumerate(rps)
        if p.tracks_begin >= 0 && p.tracks_begin < length(tracks)
            push!(result, i)
        end
    end
    return result
end

"""
    getTK_n(tracks::Vector{EDM4hep.TrackState}) -> Int

Get the number of tracks.
"""
function getTK_n(tracks::Vector{EDM4hep.TrackState})
    return length(tracks)
end

"""
    hasTRK(rps::Vector{EDM4hep.ReconstructedParticle}) -> Vector{Bool}

Check which reconstructed particles have associated tracks.
"""
function hasTRK(rps::Vector{EDM4hep.ReconstructedParticle})
    result = Bool[]
    for p in rps
        push!(result, p.tracks_begin >= 0 && p.tracks_begin != p.tracks_end)
    end
    return result
end

end # module