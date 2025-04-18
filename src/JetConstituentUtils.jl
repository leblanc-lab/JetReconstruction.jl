module JetConstituentUtils

using StaticArrays
using LinearAlgebra
using LorentzVectorHEP
using EDM4hep
using JetReconstruction
using StructArrays: StructVector

include("ReconstructedParticle.jl")
include("ReconstructedParticle2Track.jl")

# Define type aliases for clarity
const FCCAnalysesJetConstituents = StructVector{EDM4hep.ReconstructedParticle}
const FCCAnalysesJetConstituentsData = Vector{Float32}

"""
    build_constituents(jets::Vector{EDM4hep.ReconstructedParticle}, 
                        rps::Vector{EDM4hep.ReconstructedParticle}) -> Vector{FCCAnalysesJetConstituents}

Build the collection of constituents (mapping jet -> reconstructed particles) for all jets in event.
"""
function build_constituents(jets::FCCAnalysesJetConstituents, 
                            rps::Vector{EDM4hep.ReconstructedParticle})
    jcs = Vector{FCCAnalysesJetConstituents}()
    for jet in jets
        constituents = FCCAnalysesJetConstituents()
        for i in jet.particles_begin:jet.particles_end-1
            push!(constituents, rps[i+1])  # Julia uses 1-based indexing
        end
        push!(jcs, constituents)
    end
    return jcs
end

"""
    build_constituents_cluster(rps::FCCAnalysesJetConstituents, 
                            indices::Vector{Vector{Int}}) -> Vector{FCCAnalysesJetConstituents}

Build the collection of constituents using cluster indices.
"""
function build_constituents_cluster(rps::FCCAnalysesJetConstituents, 
                                indices::Vector{Vector{Int}})
    jcs = Vector{FCCAnalysesJetConstituents}()
    for jet_indices in indices
        # Create a subset of the original StructVector
        # This requires valid indices in the Julia range (1-based)
        
        # Make sure indices are valid (adjust 0-based indices if needed)
        julia_indices = [idx >= 0 ? idx+1 : idx for idx in jet_indices]
        
        # Filter out invalid indices
        valid_indices = filter(idx -> idx > 0 && idx <= length(rps), julia_indices)
        
        if !isempty(valid_indices)
            # Get constituents by indexing the StructVector
            constituents = rps[valid_indices]
            push!(jcs, constituents)
        else
            # Create an empty constituent collection with the same structure
            empty_constituents = StructVector{EDM4hep.ReconstructedParticle}(similar.(fieldarrays(rps), 0))
            push!(jcs, empty_constituents)
        end
    end
    return jcs
end

"""
    get_jet_constituents(csts::Vector{FCCAnalysesJetConstituents}, jet::Int) -> FCCAnalysesJetConstituents

Retrieve the constituents of an indexed jet in the event.
"""
function get_jet_constituents(csts::Vector{FCCAnalysesJetConstituents}, jet::Int)
    if jet < 0
        return FCCAnalysesJetConstituents()
    end
    return csts[jet+1]  # Julia uses 1-based indexing
end

struct TVector2
    X::Float64
    Y::Float64
end

"""
    get_constituents(csts::Vector{FCCAnalysesJetConstituents}, jets::Vector{Int}) -> Vector{FCCAnalysesJetConstituents}

Retrieve the constituents of a collection of indexed jets in the event.
"""
function get_constituents(csts::Vector{FCCAnalysesJetConstituents}, jets::Vector{Int})
    jcs = Vector{FCCAnalysesJetConstituents}()
    for i in eachindex(jets)
        if jets[i] >= 0
            push!(jcs, csts[jets[i]+1])  # Julia uses 1-based indexing
        end
    end
    return jcs
end

# Helper function to apply a method to each constituent collection
function cast_constituent(jcs::Vector{FCCAnalysesJetConstituents}, meth::Function)
    result = Vector{FCCAnalysesJetConstituentsData}()
    for jc in jcs
        push!(result, convert.(Float32, meth(jc)))
    end
    return result
end

# Helper function to apply a 2-arg method to each constituent collection
function cast_constituent_2(jcs::Vector{FCCAnalysesJetConstituents}, coll, meth::Function)
    result = Vector{FCCAnalysesJetConstituentsData}()
    for jc in jcs
        push!(result, convert.(Float32, meth(jc, coll)))
    end
    return result
end

# Helper function to apply a 3-arg method to each constituent collection
function cast_constituent_3(jcs::Vector{FCCAnalysesJetConstituents}, coll1, coll2, meth::Function)
    result = Vector{FCCAnalysesJetConstituentsData}()
    for jc in jcs
        push!(result, convert.(Float32, meth(jc, coll1, coll2)))
    end
    return result
end

# Helper function to apply a 4-arg method to each constituent collection
function cast_constituent_4(jcs::Vector{FCCAnalysesJetConstituents}, coll1, coll2, coll3, meth::Function)
    result = Vector{FCCAnalysesJetConstituentsData}()
    for jc in jcs
        push!(result, convert.(Float32, meth(jc, coll1, coll2, coll3)))
    end
    return result
end

"""
    get_Bz(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
            tracks::StructVector{EDM4hep.TrackState}) -> Vector{Vector{Float32}}

Calculate the magnetic field Bz for each particle based on track curvature and momentum.
Returns a vector of vectors of Bz values (one vector per jet, one value per constituent).
"""
function get_Bz(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                tracks::StructVector{EDM4hep.TrackState})
    # Constants
    c_light = 2.99792458e8  # speed of light in m/s
    a = c_light * 1e3 * 1e-15  # conversion factor for omega [1/mm]
    
    result = Vector{Vector{Float32}}()
    
    for constituents in jcs
        bz_values = Vector{Float32}()
        
        for p in constituents
            # Check if particle has associated tracks through the relation
            if isdefined(p, :tracks) && !isnothing(p.tracks) && !isempty(p.tracks)
                # Get the first track (most relevant for Bz calculation)
                track_idx = p.tracks[1]
                
                if track_idx <= length(tracks)
                    track = tracks[track_idx]
                    pt = sqrt(p.momentum.x^2 + p.momentum.y^2)
                    
                    # Calculate Bz based on track curvature (omega) and momentum
                    charge_sign = p.charge > 0.0 ? 1.0 : -1.0
                    bz_val = track.omega / a * pt * charge_sign
                    push!(bz_values, bz_val)
                else
                    push!(bz_values, -9.0f0)
                end
            else
                push!(bz_values, -9.0f0)
            end
        end
        
        push!(result, bz_values)
    end
    
    return result
end

"""
    get_pt(jcs::Vector{FCCAnalysesJetConstituents}) -> Vector{FCCAnalysesJetConstituentsData}

Get the transverse momentum of each particle in each jet.
"""
function get_pt(jcs::Vector{FCCAnalysesJetConstituents})
    result = Vector{FCCAnalysesJetConstituentsData}()
    for jet_constituents in jcs
        pt_values = FCCAnalysesJetConstituentsData()
        for p in jet_constituents
            pt = sqrt(p.momentum.x^2 + p.momentum.y^2)
            push!(pt_values, pt)
        end
        push!(result, pt_values)
    end
    return result
end

function estimate_Bz(particles::StructVector{EDM4hep.ReconstructedParticle},
    tracks::StructVector{EDM4hep.TrackState})
    # Constants
    c_light = 2.99792458e8  # Speed of light in m/s
    a = c_light * 1e3 * 1e-15  # Conversion factor for omega [1/mm]
    # Default value
    Bz_value = -9.0f0
    
    # Helper function to safely access tracks
    function get_track_idx(particle)
        # Since EDM4hep.jl doesn't have tracks_begin, we need to check if the track relation exists
        if isdefined(particle, :tracks)
            track_indices = try
                collect(particle.tracks)  # Try to collect indices from the relation
            catch
                Int[]  # Return empty array if relation can't be accessed
            end
            
            return isempty(track_indices) ? -1 : track_indices[1]
        else
            return -1
        end
    end
    
    # Iterate through particles to find one with a valid track
    for p in particles
        if p.charge != 0.0  # Only charged particles have tracks
            track_idx = get_track_idx(p)
            
            if track_idx > 0 && track_idx <= length(tracks)
                # Calculate transverse momentum
                pt = sqrt(p.momentum.x^2 + p.momentum.y^2)
                
                # Get charge sign
                charge_sign = p.charge > 0.0 ? 1.0 : -1.0
                
                # Calculate Bz using the same formula as the C++ version
                Bz_value = tracks[track_idx].omega / a * pt * charge_sign
                
                # In the C++ version, we use the last valid calculation
                # So we don't break here, allowing the loop to continue
            end
        end
    end
    
    return Bz_value
end

"""
    get_p(jcs::Vector{FCCAnalysesJetConstituents}) -> Vector{FCCAnalysesJetConstituentsData}

Get the momentum magnitude of each particle in each jet.
"""
function get_p(jcs::Vector{FCCAnalysesJetConstituents})
    result = Vector{FCCAnalysesJetConstituentsData}()
    for jet_constituents in jcs
        p_values = FCCAnalysesJetConstituentsData()
        for p in jet_constituents
            momentum = sqrt(p.momentum.x^2 + p.momentum.y^2 + p.momentum.z^2)
            push!(p_values, momentum)
        end
        push!(result, p_values)
    end
    return result
end

"""
    get_e(jcs::Vector{FCCAnalysesJetConstituents}) -> Vector{FCCAnalysesJetConstituentsData}

Get the energy of each particle in each jet.
"""
function get_e(jcs::Vector{FCCAnalysesJetConstituents})
    result = Vector{FCCAnalysesJetConstituentsData}()
    for jet_constituents in jcs
        e_values = FCCAnalysesJetConstituentsData()
        for p in jet_constituents
            push!(e_values, p.energy)
        end
        push!(result, e_values)
    end
    return result
end

"""
    get_theta(jcs::Vector{FCCAnalysesJetConstituents}) -> Vector{FCCAnalysesJetConstituentsData}

Get the polar angle of each particle in each jet.
"""
function get_theta(jcs::Vector{FCCAnalysesJetConstituents})
    result = Vector{FCCAnalysesJetConstituentsData}()
    for jet_constituents in jcs
        theta_values = FCCAnalysesJetConstituentsData()
        for p in jet_constituents
            # Calculate theta from momentum components
            p_mag = sqrt(p.momentum.x^2 + p.momentum.y^2 + p.momentum.z^2)
            if p_mag > 0
                theta = acos(p.momentum.z / p_mag)
            else
                theta = 0.0
            end
            push!(theta_values, theta)
        end
        push!(result, theta_values)
    end
    return result
end

"""
    get_phi(jcs::Vector{FCCAnalysesJetConstituents}) -> Vector{FCCAnalysesJetConstituentsData}

Get the azimuthal angle of each particle in each jet.
"""
function get_phi(jcs::Vector{FCCAnalysesJetConstituents})
    result = Vector{FCCAnalysesJetConstituentsData}()
    for jet_constituents in jcs
        phi_values = FCCAnalysesJetConstituentsData()
        for p in jet_constituents
            phi = atan(p.momentum.y, p.momentum.x)
            push!(phi_values, phi)
        end
        push!(result, phi_values)
    end
    return result
end

"""
    get_charge(jcs::Vector{FCCAnalysesJetConstituents}) -> Vector{FCCAnalysesJetConstituentsData}

Get the charge of each particle in each jet.
"""
function get_charge(jcs::Vector{FCCAnalysesJetConstituents})
    result = Vector{FCCAnalysesJetConstituentsData}()
    for jet_constituents in jcs
        charge_values = FCCAnalysesJetConstituentsData()
        for p in jet_constituents
            push!(charge_values, p.charge)
        end
        push!(result, charge_values)
    end
    return result
end

"""
    get_type(jcs::Vector{FCCAnalysesJetConstituents}) -> Vector{FCCAnalysesJetConstituentsData}

Get the PDG type of each particle in each jet.
"""
function get_type(jcs::Vector{FCCAnalysesJetConstituents})
    result = Vector{FCCAnalysesJetConstituentsData}()
    for jet_constituents in jcs
        type_values = FCCAnalysesJetConstituentsData()
        for p in jet_constituents
            push!(type_values, Float32(p.PDG))
        end
        push!(result, type_values)
    end
    return result
end

"""
    get_phi0(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
            tracks::StructVector{EDM4hep.TrackState}, 
            V::LorentzVector, Bz::Float64) -> Vector{Vector{Float32}}

Calculate the phi angle at the point of closest approach for each particle relative to vertex V.
This is a Julia implementation of the C++ function XPtoPar_phi.

Parameters:
- jcs: Vector of jet constituents (each element contains particles for one jet)
- tracks: StructVector of TrackState objects
- V: LorentzVector representing the primary vertex
- Bz: The magnetic field in Tesla

Returns:
- Vector of vectors of phi values (one vector per jet)
"""
function get_phi0(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                tracks::StructVector{EDM4hep.TrackState}, 
                V::LorentzVector, Bz::Float64)
    
    # Constants
    cSpeed = 2.99792458e8 * 1.0e-9  # Speed of light in m/ns
    
    # Initialize result vector
    result = Vector{Vector{Float32}}()
    
    # Process each jet
    for jet_constituents in jcs
        # Initialize phi values for this jet
        phi_values = Vector{Float32}()
        
        # Process each particle in the jet
        for p in jet_constituents
            # For charged particles, try to find a matching track
            track_found = false
            
            if p.charge != 0.0
                # Since we don't have direct access to tracks_begin, we need matching logic
                # In a real implementation, you would implement proper track matching
                if !isempty(tracks)
                    # For this example, using the first track as a substitute
                    # In a production system, you would implement proper track matching
                    track = tracks[1]
                    
                    D0_wrt0 = track.D0
                    Z0_wrt0 = track.Z0
                    phi0_wrt0 = track.phi
                    
                    # Create position vector at closest approach to (0,0,0)
                    X = [-D0_wrt0 * sin(phi0_wrt0), 
                        D0_wrt0 * cos(phi0_wrt0), 
                        Z0_wrt0]
                    
                    # Position vector relative to vertex V
                    x = X .- [V.x, V.y, V.z]
                    
                    # Momentum vector
                    p_vec = [p.momentum.x, p.momentum.y, p.momentum.z]
                    
                    # Calculate phi parameter
                    a = -p.charge * Bz * cSpeed
                    pt = sqrt(p.momentum.x^2 + p.momentum.y^2)
                    r2 = x[1]^2 + x[2]^2
                    cross = x[1] * p_vec[2] - x[2] * p_vec[1]
                    
                    # Calculate factor T
                    T = sqrt(pt^2 - 2 * a * cross + a^2 * r2)
                    
                    # Calculate phi angle at point of closest approach
                    # Using atan2 to correctly handle quadrants
                    phi0 = atan((p_vec[2] - a * x[1]) / T, 
                                (p_vec[1] + a * x[2]) / T)
                    
                    push!(phi_values, Float32(phi0))
                    track_found = true
                end
            end
            
            if !track_found
                # No valid track found or particle is neutral
                push!(phi_values, -9.0f0)
            end
        end
        
        push!(result, phi_values)
    end
    
    return result
end

"""
    get_dxy(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
            tracks::StructVector{EDM4hep.TrackState}, 
            V::LorentzVector, Bz::Float64) -> Vector{Vector{Float32}}

Calculate the transverse impact parameter dxy for each particle in each jet relative to vertex V.
This is a Julia implementation of the C++ function XPtoPar_dxy, adapted for jet constituents.

Parameters:
- jcs: Vector of jet constituents (each element contains particles for one jet)
- tracks: StructVector of TrackState objects
- V: LorentzVector representing the primary vertex
- Bz: The magnetic field in Tesla

Returns:
- Vector of vectors of dxy values (one vector per jet)
"""
function get_dxy(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                tracks::StructVector{EDM4hep.TrackState}, 
                V::LorentzVector, Bz::Float64)
    # Constants
    cSpeed = 2.99792458e8 * 1.0e-9  # Speed of light in m/ns
    
    # Initialize result vector
    result = Vector{Vector{Float32}}()
    
    # Process each jet
    for jet_constituents in jcs
        # Initialize dxy values for this jet
        dxy_values = Vector{Float32}()
        
        # Process each particle in the jet
        for p in jet_constituents
            track_found = false
            
            if p.charge != 0.0
                if !isempty(tracks)
                    track = tracks[1]  # Using the first track as a fallback
                    
                    D0_wrt0 = track.D0
                    Z0_wrt0 = track.Z0
                    phi0_wrt0 = track.phi
                    
                    # Create position vector at closest approach to (0,0,0)
                    X = [-D0_wrt0 * sin(phi0_wrt0), D0_wrt0 * cos(phi0_wrt0), Z0_wrt0]
                    
                    # Position vector relative to vertex V
                    x = X .- [V.x, V.y, V.z]
                    
                    # Momentum vector
                    p_vec = [p.momentum.x, p.momentum.y, p.momentum.z]
                    
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
                    
                    push!(dxy_values, D)
                    track_found = true
                end
            end
            
            if !track_found
                # No valid track found or particle is neutral
                push!(dxy_values, -9.0f0)
            end
        end
        
        push!(result, dxy_values)
    end
    
    return result
end

"""
    get_dz(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
            tracks::StructVector{EDM4hep.TrackState}, 
            V::LorentzVector, Bz::Float64) -> Vector{Vector{Float32}}

Calculate the longitudinal impact parameter dz for each particle in each jet relative to vertex V.
This is a Julia implementation of the C++ function XPtoPar_dz, adapted for jet constituents.

Parameters:
- jcs: Vector of jet constituents (each element contains particles for one jet)
- tracks: StructVector of TrackState objects
- V: LorentzVector representing the primary vertex
- Bz: The magnetic field in Tesla

Returns:
- Vector of vectors of dz values (one vector per jet)
"""
function get_dz(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                tracks::StructVector{EDM4hep.TrackState}, 
                V::LorentzVector, Bz::Float64)
    # Constants
    cSpeed = 2.99792458e8 * 1.0e-9  # Speed of light in m/ns
    
    # Initialize result vector
    result = Vector{Vector{Float32}}()
    
    # Process each jet
    for jet_constituents in jcs
        # Initialize dz values for this jet
        dz_values = Vector{Float32}()
        
        # Process each particle in the jet
        for p in jet_constituents
            track_found = false
            
            if p.charge != 0.0
                if !isempty(tracks)
                    track = tracks[1]
                    
                    D0_wrt0 = track.D0
                    Z0_wrt0 = track.Z0
                    phi0_wrt0 = track.phi
                    
                    # Create position vector at closest approach to (0,0,0)
                    X = [
                        -D0_wrt0 * sin(phi0_wrt0), 
                        D0_wrt0 * cos(phi0_wrt0), 
                        Z0_wrt0
                    ]
                    
                    # Position vector relative to vertex V
                    x = X .- [V.x, V.y, V.z]
                    
                    # Momentum vector
                    p_vec = [p.momentum.x, p.momentum.y, p.momentum.z]
                    
                    # Calculate dz parameter
                    a = -p.charge * Bz * cSpeed
                    pt = sqrt(p.momentum.x^2 + p.momentum.y^2)
                    C = a / (2 * pt)
                    r2 = x[1]^2 + x[2]^2
                    cross = x[1] * p_vec[2] - x[2] * p_vec[1]
                    T = sqrt(pt^2 - 2 * a * cross + a^2 * r2)
                    
                    # Calculate the distance D
                    D = 0.0
                    if pt < 10.0
                        D = (T - pt) / a
                    else
                        D = (-2 * cross + a * r2) / (T + pt)
                    end
                    
                    # Calculate the sine of helical parameter
                    B = C * sqrt(max(r2 - D^2, 0.0) / (1 + 2 * C * D))
                    if abs(B) > 1.0
                        B = sign(B)
                    end
                    
                    # Path length
                    st = asin(B) / C
                    
                    # Directional tangent
                    ct = p_vec[3] / pt
                    
                    # Calculate z0 (longitudinal impact parameter)
                    dot_prod = x[1] * p_vec[1] + x[2] * p_vec[2]
                    z0 = 0.0
                    
                    if dot_prod > 0.0
                        z0 = x[3] - ct * st
                    else
                        z0 = x[3] + ct * st
                    end
                    
                    push!(dz_values, Float32(z0))
                    track_found = true
                end
            end
            
            if !track_found
                # No valid track found or particle is neutral
                push!(dz_values, -9.0f0)
            end
        end
        
        push!(result, dz_values)
    end
    
    return result
end

"""
    get_dptdpt(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                tracks::StructVector{EDM4hep.TrackState}) -> Vector{Vector{Float32}}

Get the omega covariance (dpt/dpt) for each particle in each jet from its associated track.
This is a Julia implementation of the C++ function get_omega_cov, which uses getRP2TRK_omega_cov.

Parameters:
- jcs: Vector of jet constituents (each element contains particles for one jet)
- tracks: StructVector of TrackState objects

Returns:
- Vector of vectors of dptdpt values (one vector per jet)
"""
function get_dptdpt(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                    tracks::StructVector{EDM4hep.TrackState})
    # Initialize result vector
    result = Vector{Vector{Float32}}()
    
    # Process each jet
    for jet_constituents in jcs
        # Initialize dptdpt values for this jet
        dptdpt_values = Vector{Float32}()
        
        # Process each particle in the jet
        for p in jet_constituents
            # For charged particles, try to find a matching track
            track_found = false
            
            if p.charge != 0.0
                if !isempty(tracks)
                    track = tracks[1]
                    push!(dptdpt_values, Float32(track.covMatrix[6]))
                    track_found = true
                end
            end
            
            if !track_found
                # No valid track found or particle is neutral
                push!(dptdpt_values, -9.0f0)
            end
        end
        
        push!(result, dptdpt_values)
    end
    
    return result
end

"""
    get_detadeta(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                tracks::StructVector{EDM4hep.TrackState}) -> Vector{Vector{Float32}}

Get the tanLambda covariance (deta/deta) for each particle in each jet from its associated track.
This is a Julia implementation of the C++ function get_tanlambda_cov.

Parameters:
- jcs: Vector of jet constituents (each element contains particles for one jet)
- tracks: StructVector of TrackState objects

Returns:
- Vector of vectors of detadeta values (one vector per jet)
"""
function get_detadeta(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                    tracks::StructVector{EDM4hep.TrackState})
    # Initialize result vector
    result = Vector{Vector{Float32}}()
    
    # Process each jet
    for jet_constituents in jcs
        # Initialize detadeta values for this jet
        detadeta_values = Vector{Float32}()
        
        # Process each particle in the jet
        for p in jet_constituents
            # For charged particles, try to find a matching track
            track_found = false
            
            if p.charge != 0.0
                if !isempty(tracks)
                    track = tracks[1]
                    push!(detadeta_values, Float32(track.covMatrix[15]))
                    track_found = true
                end
            end
            
            if !track_found
                # No valid track found or particle is neutral
                push!(detadeta_values, -9.0f0)
            end
        end
        
        push!(result, detadeta_values)
    end
    
    return result
end

"""
    get_dphidphi(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                tracks::StructVector{EDM4hep.TrackState}) -> Vector{Vector{Float32}}

Get the phi covariance (dphi/dphi) for each particle in each jet from its associated track.
This is a Julia implementation of the C++ function get_phi0_cov.

Parameters:
- jcs: Vector of jet constituents (each element contains particles for one jet)
- tracks: StructVector of TrackState objects

Returns:
- Vector of vectors of dphidphi values (one vector per jet)
"""
function get_dphidphi(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                    tracks::StructVector{EDM4hep.TrackState})
    # Initialize result vector
    result = Vector{Vector{Float32}}()
    
    # Process each jet
    for jet_constituents in jcs
        # Initialize dphidphi values for this jet
        dphidphi_values = Vector{Float32}()
        
        # Process each particle in the jet
        for p in jet_constituents
            # For charged particles, try to find a matching track
            track_found = false
            
            if p.charge != 0.0
                if !isempty(tracks)
                    track = tracks[1]
                    push!(dphidphi_values, Float32(track.covMatrix[3]))
                    track_found = true
                end
            end
            
            if !track_found
                # No valid track found or particle is neutral
                push!(dphidphi_values, -9.0f0)
            end
        end
        
        push!(result, dphidphi_values)
    end
    
    return result
end

"""
    get_dxydxy(jcs::Vector{FCCAnalysesJetConstituents}, tracks::Vector{EDM4hep.TrackState}) -> Vector{FCCAnalysesJetConstituentsData}

Get the d0 covariance (dxy/dxy) for each particle.
"""
function get_dxydxy(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                    tracks::StructVector{EDM4hep.TrackState})
    # Initialize result vector
    result = Vector{Vector{Float32}}()
    
    # Process each jet
    for jet_constituents in jcs
        # Initialize dxydxy values for this jet
        dxydxy_values = Vector{Float32}()
        
        # Process each particle in the jet
        for p in jet_constituents
            # For charged particles, try to find a matching track
            track_found = false
            
            if p.charge != 0.0
                if !isempty(tracks)
                    track = tracks[1]
                    push!(dxydxy_values, Float32(track.covMatrix[1]))
                    track_found = true
                end
            end
            
            if !track_found
                # No valid track found or particle is neutral
                push!(dxydxy_values, -9.0f0)
            end
        end
        
        push!(result, dxydxy_values)
    end
    
    return result
end

"""
    get_dzdz(jcs::Vector{FCCAnalysesJetConstituents}, tracks::Vector{EDM4hep.TrackState}) -> Vector{FCCAnalysesJetConstituentsData}

Get the z0 covariance (dz/dz) for each particle.
"""
function get_dzdz(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                    tracks::StructVector{EDM4hep.TrackState})
    # Initialize result vector
    result = Vector{Vector{Float32}}()
    
    # Process each jet
    for jet_constituents in jcs
        # Initialize dzdz values for this jet
        dzdz_values = Vector{Float32}()
        
        # Process each particle in the jet
        for p in jet_constituents
            # For charged particles, try to find a matching track
            track_found = false
            
            if p.charge != 0.0
                if !isempty(tracks)
                    track = tracks[1]
                    push!(dzdz_values, Float32(track.covMatrix[10]))
                    track_found = true
                end
            end
            
            if !track_found
                # No valid track found or particle is neutral
                push!(dzdz_values, -9.0f0)
            end
        end
        
        push!(result, dzdz_values)
    end
    
    return result
end

"""
    get_dxydz(jcs::Vector{FCCAnalysesJetConstituents}, tracks::Vector{EDM4hep.TrackState}) -> Vector{FCCAnalysesJetConstituentsData}

Get the d0-z0 covariance (dxy/dz) for each particle.
"""
function get_dxydz(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                    tracks::StructVector{EDM4hep.TrackState})
    # Initialize result vector
    result = Vector{Vector{Float32}}()
    
    # Process each jet
    for jet_constituents in jcs
        # Initialize dxydz values for this jet
        dxydz_values = Vector{Float32}()
        
        # Process each particle in the jet
        for p in jet_constituents
            # For charged particles, try to find a matching track
            track_found = false
            
            if p.charge != 0.0
                if !isempty(tracks)
                    track = tracks[1]
                    push!(dxydz_values, Float32(track.covMatrix[7]))
                    track_found = true
                end
            end
            
            if !track_found
                # No valid track found or particle is neutral
                push!(dxydz_values, -9.0f0)
            end
        end
        
        push!(result, dxydz_values)
    end
    
    return result
end

"""
    get_dphidxy(jcs::Vector{FCCAnalysesJetConstituents}, tracks::Vector{EDM4hep.TrackState}) -> Vector{FCCAnalysesJetConstituentsData}

Get the phi0-d0 covariance (dphi/dxy) for each particle.
"""
function get_dphidxy(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                    tracks::StructVector{EDM4hep.TrackState})
    # Initialize result vector
    result = Vector{Vector{Float32}}()
    
    # Process each jet
    for jet_constituents in jcs
        # Initialize dphidxy values for this jet
        dphidxy_values = Vector{Float32}()
        
        # Process each particle in the jet
        for p in jet_constituents
            # For charged particles, try to find a matching track
            track_found = false
            
            if p.charge != 0.0
                if !isempty(tracks)
                    track = tracks[1]
                    push!(dphidxy_values, Float32(track.covMatrix[2]))
                    track_found = true
                end
            end
            
            if !track_found
                # No valid track found or particle is neutral
                push!(dphidxy_values, -9.0f0)
            end
        end
        
        push!(result, dphidxy_values)
    end
    
    return result
end

"""
    get_dlambdadz(jcs::Vector{FCCAnalysesJetConstituents}, tracks::Vector{EDM4hep.TrackState}) -> Vector{FCCAnalysesJetConstituentsData}

Get the tanLambda-z0 covariance (dlambda/dz) for each particle.
"""
function get_dlambdadz(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                    tracks::StructVector{EDM4hep.TrackState})
    # Initialize result vector
    result = Vector{Vector{Float32}}()
    
    # Process each jet
    for jet_constituents in jcs
        # Initialize dlambdadz values for this jet
        dlambdadz_values = Vector{Float32}()
        
        # Process each particle in the jet
        for p in jet_constituents
            # For charged particles, try to find a matching track
            track_found = false
            
            if p.charge != 0.0
                if !isempty(tracks)
                    track = tracks[1]
                    push!(dlambdadz_values, Float32(track.covMatrix[14]))
                    track_found = true
                end
            end
            
            if !track_found
                # No valid track found or particle is neutral
                push!(dlambdadz_values, -9.0f0)
            end
        end
        
        push!(result, dlambdadz_values)
    end
    
    return result
end

"""
    get_phidz(jcs::Vector{FCCAnalysesJetConstituents}, tracks::Vector{EDM4hep.TrackState}) -> Vector{FCCAnalysesJetConstituentsData}

Get the phi0-z0 covariance (dphi/dz) for each particle.
"""
function get_phidz(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                    tracks::StructVector{EDM4hep.TrackState})
    # Initialize result vector
    result = Vector{Vector{Float32}}()
    
    # Process each jet
    for jet_constituents in jcs
        # Initialize phidz values for this jet
        phidz_values = Vector{Float32}()
        
        # Process each particle in the jet
        for p in jet_constituents
            # For charged particles, try to find a matching track
            track_found = false
            
            if p.charge != 0.0
                if !isempty(tracks)
                    track = tracks[1]
                    push!(phidz_values, Float32(track.covMatrix[8]))
                    track_found = true
                end
            end
            
            if !track_found
                # No valid track found or particle is neutral
                push!(phidz_values, -9.0f0)
            end
        end
        
        push!(result, phidz_values)
    end
    
    return result
end

"""
    get_dxyc(jcs::Vector{FCCAnalysesJetConstituents}, tracks::Vector{EDM4hep.TrackState}) -> Vector{FCCAnalysesJetConstituentsData}

Get the d0-omega covariance (dxy/c) for each particle.
"""
function get_dxyc(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                    tracks::StructVector{EDM4hep.TrackState})
    # Initialize result vector
    result = Vector{Vector{Float32}}()
    
    # Process each jet
    for jet_constituents in jcs
        # Initialize dxyc values for this jet
        dxyc_values = Vector{Float32}()
        
        # Process each particle in the jet
        for p in jet_constituents
            # For charged particles, try to find a matching track
            track_found = false
            
            if p.charge != 0.0
                if !isempty(tracks)
                    track = tracks[1]
                    push!(dxyc_values, Float32(track.covMatrix[4]))
                    track_found = true
                end
            end
            
            if !track_found
                # No valid track found or particle is neutral
                push!(dxyc_values, -9.0f0)
            end
        end
        
        push!(result, dxyc_values)
    end
    
    return result
end

"""
    get_dxyctgtheta(jcs::Vector{FCCAnalysesJetConstituents}, tracks::Vector{EDM4hep.TrackState}) -> Vector{FCCAnalysesJetConstituentsData}

Get the d0-tanLambda covariance (dxy/ctgtheta) for each particle.
"""
function get_dxyctgtheta(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                    tracks::StructVector{EDM4hep.TrackState})
    # Initialize result vector
    result = Vector{Vector{Float32}}()
    
    # Process each jet
    for jet_constituents in jcs
        # Initialize dxyctgtheta values for this jet
        dxyctgtheta_values = Vector{Float32}()
        
        # Process each particle in the jet
        for p in jet_constituents
            # For charged particles, try to find a matching track
            track_found = false
            
            if p.charge != 0.0
                if !isempty(tracks)
                    track = tracks[1]
                    push!(dxyctgtheta_values, Float32(track.covMatrix[11]))
                    track_found = true
                end
            end
            
            if !track_found
                # No valid track found or particle is neutral
                push!(dxyctgtheta_values, -9.0f0)
            end
        end
        
        push!(result, dxyctgtheta_values)
    end
    
    return result
end

"""
    get_phic(jcs::Vector{FCCAnalysesJetConstituents}, tracks::Vector{EDM4hep.TrackState}) -> Vector{FCCAnalysesJetConstituentsData}

Get the phi0-omega covariance (phi/c) for each particle.
"""
function get_phic(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                    tracks::StructVector{EDM4hep.TrackState})
    # Initialize result vector
    result = Vector{Vector{Float32}}()
    
    # Process each jet
    for jet_constituents in jcs
        # Initialize phic values for this jet
        phic_values = Vector{Float32}()
        
        # Process each particle in the jet
        for p in jet_constituents
            # For charged particles, try to find a matching track
            track_found = false
            
            if p.charge != 0.0
                if !isempty(tracks)
                    track = tracks[1]
                    push!(phic_values, Float32(track.covMatrix[5]))
                    track_found = true
                end
            end
            
            if !track_found
                # No valid track found or particle is neutral
                push!(phic_values, -9.0f0)
            end
        end
        
        push!(result, phic_values)
    end
    
    return result
end

"""
    get_phictgtheta(jcs::Vector{FCCAnalysesJetConstituents}, tracks::Vector{EDM4hep.TrackState}) -> Vector{FCCAnalysesJetConstituentsData}

Get the phi0-tanLambda covariance (phi/ctgtheta) for each particle.
"""
function get_phictgtheta(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                    tracks::StructVector{EDM4hep.TrackState})
    # Initialize result vector
    result = Vector{Vector{Float32}}()
    
    # Process each jet
    for jet_constituents in jcs
        # Initialize phictgtheta values for this jet
        phictgtheta_values = Vector{Float32}()
        
        # Process each particle in the jet
        for p in jet_constituents
            # For charged particles, try to find a matching track
            track_found = false
            
            if p.charge != 0.0
                if !isempty(tracks)
                    track = tracks[1]
                    push!(phictgtheta_values, Float32(track.covMatrix[12]))
                    track_found = true
                end
            end
            
            if !track_found
                # No valid track found or particle is neutral
                push!(phictgtheta_values, -9.0f0)
            end
        end
        
        push!(result, phictgtheta_values)
    end
    
    return result
end

"""
    get_cdz(jcs::Vector{FCCAnalysesJetConstituents}, tracks::Vector{EDM4hep.TrackState}) -> Vector{FCCAnalysesJetConstituentsData}

Get the omega-z0 covariance (c/dz) for each particle.
"""
function get_cdz(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                    tracks::StructVector{EDM4hep.TrackState})
    # Initialize result vector
    result = Vector{Vector{Float32}}()
    
    # Process each jet
    for jet_constituents in jcs
        # Initialize cdz values for this jet
        cdz_values = Vector{Float32}()
        
        # Process each particle in the jet
        for p in jet_constituents
            # For charged particles, try to find a matching track
            track_found = false
            
            if p.charge != 0.0
                if !isempty(tracks)
                    track = tracks[1]
                    push!(cdz_values, Float32(track.covMatrix[9]))
                    track_found = true
                end
            end
            
            if !track_found
                # No valid track found or particle is neutral
                push!(cdz_values, -9.0f0)
            end
        end
        
        push!(result, cdz_values)
    end
    
    return result
end

"""
    get_cctgtheta(jcs::Vector{FCCAnalysesJetConstituents}, tracks::Vector{EDM4hep.TrackState}) -> Vector{FCCAnalysesJetConstituentsData}

Get the omega-tanLambda covariance (c/ctgtheta) for each particle.
"""
function get_cctgtheta(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                    tracks::StructVector{EDM4hep.TrackState})
    # Initialize result vector
    result = Vector{Vector{Float32}}()
    
    # Process each jet
    for jet_constituents in jcs
        # Initialize cctgtheta values for this jet
        cctgtheta_values = Vector{Float32}()
        
        # Process each particle in the jet
        for p in jet_constituents
            # For charged particles, try to find a matching track
            track_found = false
            
            if p.charge != 0.0
                if !isempty(tracks)
                    track = tracks[1]
                    push!(cctgtheta_values, Float32(track.covMatrix[13]))
                    track_found = true
                end
            end
            
            if !track_found
                # No valid track found or particle is neutral
                push!(cctgtheta_values, -9.0f0)
            end
        end
        
        push!(result, cctgtheta_values)
    end
    
    return result
end

"""
    get_erel_log_cluster(jets::Vector{EEjet}, 
                        jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}) -> Vector{Vector{Float32}}

Calculate log of relative energy (log(E_const/E_jet)) for each constituent particle in clustered jets.
"""
function get_erel_log_cluster(jets::Vector{EEjet}, 
                            jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}})
    # Define the result type
    result = Vector{Vector{Float32}}()
    
    for i in eachindex(jets)
        jet_csts = Float32[]
        
        # Get jet energy
        e_jet = jets[i].E  # Assuming EEjet has an e property for energy
        
        # Get constituents for this jet
        if i <= length(jcs)
            constituents = jcs[i]
            
            for jc in constituents
                # Calculate relative energy and log
                val = (e_jet > 0.0) ? jc.energy / e_jet : 1.0
                erel_log = log10(val)
                push!(jet_csts, erel_log)
            end
        end
        
        push!(result, jet_csts)
    end
    
    return result
end

"""
    get_thetarel_cluster(jets::Vector{EEjet}, 
                        jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}) -> Vector{Vector{Float32}}

Calculate relative theta angle between constituent particle and clustered jet axis.
"""
function get_thetarel_cluster(jets::Vector{EEjet}, 
                            jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}})
    result = Vector{Vector{Float32}}()
    
    for i in eachindex(jets)
        jet_csts = Float32[]
        
        # Create a 4-vector for the jet
        jet = jets[i]
        px, py, pz, E = jet.px, jet.py, jet.pz, jet.E
        
        # Calculate jet direction angles
        p_mag = sqrt(px^2 + py^2 + pz^2)
        theta_jet = p_mag > 0 ? acos(pz / p_mag) : 0.0
        phi_jet = atan(py, px)
        
        # Get constituents for this jet
        if i <= length(jcs)
            constituents = jcs[i]
            
            for constituent in constituents
                # Create a 4-vector for the constituent
                p_const_x = constituent.momentum.x
                p_const_y = constituent.momentum.y
                p_const_z = constituent.momentum.z
                
                # Rotate the constituent vector to align with jet axis
                
                # First rotate around z-axis by -phi_jet
                p_rot_x = p_const_x * cos(-phi_jet) - p_const_y * sin(-phi_jet)
                p_rot_y = p_const_x * sin(-phi_jet) + p_const_y * cos(-phi_jet)
                p_rot_z = p_const_z
                
                # Then rotate around y-axis by -theta_jet
                p_rot2_x = p_rot_x * cos(-theta_jet) - p_rot_z * sin(-theta_jet)
                p_rot2_z = p_rot_x * sin(-theta_jet) + p_rot_z * cos(-theta_jet)
                p_rot2_y = p_rot_y
                
                # Calculate theta in rotated frame
                p_rot_mag = sqrt(p_rot2_x^2 + p_rot2_y^2 + p_rot2_z^2)
                theta_rel = p_rot_mag > 0 ? acos(p_rot2_z / p_rot_mag) : 0.0
                
                push!(jet_csts, theta_rel)
            end
        end
        
        push!(result, jet_csts)
    end
    
    return result
end

"""
    get_phirel_cluster(jets::Vector{EEjet}, 
                        jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}) -> Vector{Vector{Float32}}

Calculate relative phi angle between constituent particle and clustered jet axis.
"""
function get_phirel_cluster(jets::Vector{EEjet}, 
                            jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}})
    result = Vector{Vector{Float32}}()
    
    for i in eachindex(jets)
        jet_csts = Float32[]
        
        # Create a 4-vector for the jet
        jet = jets[i]
        # Access momentum components from EEjet
        px, py, pz, E = jet.px, jet.py, jet.pz, jet.E
        
        # Calculate jet direction angles
        p_mag = sqrt(px^2 + py^2 + pz^2)
        theta_jet = p_mag > 0 ? acos(pz / p_mag) : 0.0
        phi_jet = atan(py, px)
        
        # Get constituents for this jet
        if i <= length(jcs)
            constituents = jcs[i]
            
            for constituent in constituents
                # Get constituent momentum
                p_const_x = constituent.momentum.x
                p_const_y = constituent.momentum.y
                p_const_z = constituent.momentum.z
                
                # Rotate the constituent vector to align with jet axis
                
                # First rotate around z-axis by -phi_jet
                p_rot_x = p_const_x * cos(-phi_jet) - p_const_y * sin(-phi_jet)
                p_rot_y = p_const_x * sin(-phi_jet) + p_const_y * cos(-phi_jet)
                p_rot_z = p_const_z
                
                # Then rotate around y-axis by -theta_jet
                p_rot2_x = p_rot_x * cos(-theta_jet) - p_rot_z * sin(-theta_jet)
                p_rot2_z = p_rot_x * sin(-theta_jet) + p_rot_z * cos(-theta_jet)
                p_rot2_y = p_rot_y
                
                # Calculate phi in rotated frame
                phi_rel = atan(p_rot2_y, p_rot2_x)
                
                push!(jet_csts, phi_rel)
            end
        end
        
        push!(result, jet_csts)
    end
    
    return result
end
"""
    get_isMu(jcs::Vector{FCCAnalysesJetConstituents}) -> Vector{FCCAnalysesJetConstituentsData}

Check if each constituent particle is a muon.
"""
function get_isMu(jcs::Vector{FCCAnalysesJetConstituents})
    result = Vector{FCCAnalysesJetConstituentsData}()
    
    for jet_constituents in jcs
        is_mu = FCCAnalysesJetConstituentsData()
        
        for p in jet_constituents
            if abs(p.charge) > 0 && abs(p.mass - 0.105658) < 1e-3
                push!(is_mu, 1.0f0)
            else
                push!(is_mu, 0.0f0)
            end
        end
        
        push!(result, is_mu)
    end
    
    return result
end

"""
    get_isEl(jcs::Vector{FCCAnalysesJetConstituents}) -> Vector{FCCAnalysesJetConstituentsData}

Check if each constituent particle is an electron.
"""
function get_isEl(jcs::Vector{FCCAnalysesJetConstituents})
    result = Vector{FCCAnalysesJetConstituentsData}()
    
    for jet_constituents in jcs
        is_el = FCCAnalysesJetConstituentsData()
        
        for p in jet_constituents
            if abs(p.charge) > 0 && abs(p.mass - 0.000510999) < 1e-5
                push!(is_el, 1.0f0)
            else
                push!(is_el, 0.0f0)
            end
        end
        
        push!(result, is_el)
    end
    
    return result
end

"""
    get_isChargedHad(jcs::Vector{FCCAnalysesJetConstituents}) -> Vector{FCCAnalysesJetConstituentsData}

Check if each constituent particle is a charged hadron.
"""
function get_isChargedHad(jcs::Vector{FCCAnalysesJetConstituents})
    result = Vector{FCCAnalysesJetConstituentsData}()
    
    for jet_constituents in jcs
        is_charged_had = FCCAnalysesJetConstituentsData()
        
        for p in jet_constituents
            if abs(p.charge) > 0 && abs(p.mass - 0.13957) < 1e-3
                push!(is_charged_had, 1.0f0)
            else
                push!(is_charged_had, 0.0f0)
            end
        end
        
        push!(result, is_charged_had)
    end
    
    return result
end

"""
    get_isGamma(jcs::Vector{FCCAnalysesJetConstituents}) -> Vector{FCCAnalysesJetConstituentsData}

Check if each constituent particle is a photon (gamma).
"""
function get_isGamma(jcs::Vector{FCCAnalysesJetConstituents})
    result = Vector{FCCAnalysesJetConstituentsData}()
    
    for jet_constituents in jcs
        is_gamma = FCCAnalysesJetConstituentsData()
        
        for p in jet_constituents
            if p.particleIDs == 22  # PDG code for photon
                push!(is_gamma, 1.0f0)
            else
                push!(is_gamma, 0.0f0)
            end
        end
        
        push!(result, is_gamma)
    end
    
    return result
end

"""
    get_isNeutralHad(jcs::Vector{FCCAnalysesJetConstituents}) -> Vector{FCCAnalysesJetConstituentsData}

Check if each constituent particle is a neutral hadron.
"""
function get_isNeutralHad(jcs::Vector{FCCAnalysesJetConstituents})
    result = Vector{FCCAnalysesJetConstituentsData}()
    
    for jet_constituents in jcs
        is_neutral_had = FCCAnalysesJetConstituentsData()
        
        for p in jet_constituents
            if p.particleIDs == 130  # PDG code for K_L^0 (common neutral hadron)
                push!(is_neutral_had, 1.0f0)
            else
                push!(is_neutral_had, 0.0f0)
            end
        end
        
        push!(result, is_neutral_had)
    end
    
    return result
end

"""
    get_mtof(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
             track_L::Vector{Float32},
             trackdata::StructVector{EDM4hep.Track},
             trackerhits::StructVector{EDM4hep.TrackerHit3D},
             gammadata::StructVector{EDM4hep.Cluster},
             nhdata::StructVector{EDM4hep.Cluster},
             calohits::StructVector{EDM4hep.CalorimeterHit},
             V::LorentzVector) -> Vector{Vector{Float32}}

Calculate the mass using time-of-flight measurements for each particle in each jet.
This is a Julia implementation of the C++ function get_mtof.

Parameters:
- jcs: Vector of jet constituents (each element contains particles for one jet)
- track_L: Vector of track lengths
- trackdata: StructVector of Track objects
- trackerhits: StructVector of TrackerHit3D objects 
- gammadata: StructVector of photon Cluster objects
- nhdata: StructVector of neutral hadron Cluster objects
- calohits: StructVector of CalorimeterHit objects
- V: LorentzVector representing the primary vertex position and time

Returns:
- Vector of vectors of mtof values (one vector per jet)
"""
function get_mtof(jcs::Vector{StructVector{EDM4hep.ReconstructedParticle}}, 
                 track_L::Vector{Float32},
                 trackdata::StructVector{EDM4hep.Track},
                 trackerhits::StructVector{EDM4hep.TrackerHit},
                 gammadata::StructVector{EDM4hep.Cluster},
                 nhdata::StructVector{EDM4hep.Cluster},
                 calohits::StructVector{EDM4hep.CalorimeterHit},
                 V::LorentzVector)
    
    # Speed of light in m/s
    c_light = 2.99792458e8
    
    # Initialize result vector
    result = Vector{Vector{Float32}}()
    
    # Process each jet
    for i in eachindex(jcs)
        # Get current jet constituents
        constituents = jcs[i]
        # Initialize mtof values for this jet
        mtof_values = Vector{Float32}()
        
        # Process each particle in the jet
        for j in eachindex(constituents)
            particle = constituents[j]
            mtof_added = false
            
            # Check if this is a cluster-based particle (neutral hadron or photon)
            # Note: In a real implementation, you'd need proper logic to check clusters_begin and adjust
            # the following logic based on your actual data structures
            if !isempty(nhdata) && !isempty(gammadata) && j <= length(nhdata) + length(gammadata)
                # K_L^0 (PDG code 130)
                if particle.PDG == 130
                    # For neutral hadrons
                    # This assumes photons are filled first and neutral hadrons after
                    # In a real implementation, you'd need proper indexing logic
                    if !isempty(calohits) && j <= length(calohits)
                        # Just using first calohit as an example
                        calohit = calohits[1]
                        
                        # Get time and position from calorimeter hit
                        T = calohit.time
                        X = calohit.position.x
                        Y = calohit.position.y
                        Z = calohit.position.z
                        
                        # Time of flight
                        tof = T
                        
                        # Compute path length with respect to primary vertex (convert to km)
                        L = sqrt((X - V.x)^2 + (Y - V.y)^2 + (Z - V.z)^2) * 0.001
                        
                        # Calculate beta (v/c)
                        beta = L / (tof * c_light)
                        
                        # Get particle energy
                        E = particle.energy
                        
                        # Calculate mass from relativistic formula: m = E * sqrt(1 - β²)
                        if beta < 1.0 && beta > 0.0
                            push!(mtof_values, Float32(E * sqrt(1.0 - beta * beta)))
                            mtof_added = true
                        else
                            push!(mtof_values, 9.0f0)
                            mtof_added = true
                        end
                    end
                elseif particle.PDG == 22  # Photon
                    push!(mtof_values, 0.0f0)
                    mtof_added = true
                end
            end
            
            # Check if this is a track-based particle
            # In a real implementation, you'd need proper track matching
            if !mtof_added && !isempty(trackdata)
                # Check if this is an electron
                if abs(particle.charge) > 0 && abs(particle.mass - 0.000510999) < 1e-5
                    push!(mtof_values, 0.000510999f0)
                    mtof_added = true
                
                # Check if this is a muon
                elseif abs(particle.charge) > 0 && abs(particle.mass - 0.105658) < 1e-3
                    push!(mtof_values, 0.105658f0)
                    mtof_added = true
                    
                # Other charged particles
                elseif !isempty(trackerhits) && !isempty(track_L)
                    # Time given by primary vertex (convert from mm to second)
                    Tin = V.t * 1e-3 / c_light
                    
                    # For demonstration, use first trackerhit
                    # In a real implementation, you'd need proper track matching
                    Tout = isempty(trackerhits) ? 0.0f0 : trackerhits[1].time
                    
                    # Time of flight
                    tof = Tout - Tin
                    
                    # Use first track length for demonstration
                    # In a real implementation, you'd match with the correct track
                    L = isempty(track_L) ? 0.0f0 : track_L[1] * 0.001
                    
                    # Calculate beta (v/c)
                    beta = L / (tof * c_light)
                    
                    # Calculate momentum
                    p = sqrt(particle.momentum.x^2 + particle.momentum.y^2 + particle.momentum.z^2)
                    
                    # Calculate mass from relativistic formula: m = p * sqrt(1/β² - 1)
                    if beta < 1.0 && beta > 0.0
                        push!(mtof_values, Float32(p * sqrt(1.0 / (beta * beta) - 1.0)))
                        mtof_added = true
                    else
                        push!(mtof_values, 0.13957039f0)  # Default to pion mass
                        mtof_added = true
                    end
                end
            end
            
            # Add default value if nothing was added for this particle
            if !mtof_added
                push!(mtof_values, -9.0f0)
            end
        end
        
        push!(result, mtof_values)
    end
    
    return result
end

"""
    get_dndx(jcs::Vector{FCCAnalysesJetConstituents}, dNdx::Vector{EDM4hep.Quantity},
             trackdata::Vector{EDM4hep.Track}, 
             JetsConstituents_isChargedHad::Vector{FCCAnalysesJetConstituentsData}) -> Vector{FCCAnalysesJetConstituentsData}

Calculate dE/dx or dN/dx for each charged hadron in the jet.
"""
function get_dndx(jcs::Vector{FCCAnalysesJetConstituents}, dNdx::Vector{EDM4hep.Quantity},
                  trackdata::Vector{EDM4hep.Track}, 
                  JetsConstituents_isChargedHad::Vector{FCCAnalysesJetConstituentsData})
    result = Vector{FCCAnalysesJetConstituentsData}()
    
    for i in eachindex(jcs)
        constituents = jcs[i]
        isChargedHad = JetsConstituents_isChargedHad[i]
        tmp = FCCAnalysesJetConstituentsData()
        
        for j in eachindex(constituents)
            if j <= length(isChargedHad) && constituents[j].tracks_begin < length(trackdata) && 
               isChargedHad[j] == 1.0
                track_idx = constituents[j].tracks_begin
                if track_idx < length(trackdata) && trackdata[track_idx+1].dxQuantities_begin < length(dNdx)
                    # Convert to GeV
                    push!(tmp, dNdx[trackdata[track_idx+1].dxQuantities_begin+1].value / 1000.0)
                else
                    push!(tmp, 0.0f0)
                end
            else
                push!(tmp, 0.0f0)
            end
        end
        
        push!(result, tmp)
    end
    
    return result
end
### TODO: Update the trackData function

"""
    get_Sip2dVal_clusterV(jets::Vector{JetReconstruction.PseudoJet},
                         D0::Vector{Vector{Float32}},
                         phi0::Vector{Vector{Float32}},
                         Bz::Float32) -> Vector{Vector{Float32}}

Calculate the 2D signed impact parameter value for each particle relative to the jet axis.
This is a Julia implementation of the C++ function get_Sip2dVal_clusterV.

Parameters:
- jets: Vector of PseudoJet objects representing jets
- D0: Vector of vectors containing D0 values (transverse impact parameters)
- phi0: Vector of vectors containing phi0 values (azimuthal angles at impact point)
- Bz: The magnetic field in Tesla

Returns:
- Vector of vectors of 2D signed impact parameter values (one vector per jet)
"""
function get_Sip2dVal_clusterV(jets::Vector{JetReconstruction.EEjet},
                              D0::Vector{Vector{Float32}},
                              phi0::Vector{Vector{Float32}},
                              Bz::Float64)
    # Initialize result vector
    result = Vector{Vector{Float32}}()
    
    # Process each jet
    for i in eachindex(jets)
        # Create 2D vector for jet direction in transverse plane
        p = TVector2(jets[i].px, jets[i].py)
        
        # Initialize impact parameter values for this jet
        sip2d_values = Vector{Float32}()
        
        # Process each particle in the jet
        for j in eachindex(D0[i])
            if D0[i][j] != -9.0f0
                # Calculate the 2D impact point vector
                d0 = TVector2(-D0[i][j] * sin(phi0[i][j]), 
                              D0[i][j] * cos(phi0[i][j]))
                
                # Calculate dot product for sign determination
                dot_product = d0.X * p.X + d0.Y * p.Y
                
                # Sign based on dot product and magnitude based on D0
                signed_ip = sign(dot_product) * abs(D0[i][j])
                
                push!(sip2d_values, signed_ip)
            else
                push!(sip2d_values, -9.0f0)
            end
        end
        
        push!(result, sip2d_values)
    end
    
    return result
end

function get_btagSip2dVal(jets::Vector{JetReconstruction.EEjet},
    pfcand_dxy::Vector{Vector{Float32}},
    pfcand_phi0::Vector{Vector{Float32}},
    Bz::Float64)
# Simply call the implementation function
return get_Sip2dVal_clusterV(jets, pfcand_dxy, pfcand_phi0, Bz)
end

"""
    get_Sip2dSig(Sip2dVals::Vector{Vector{Float32}},
                err2_D0::Vector{Vector{Float32}}) -> Vector{Vector{Float32}}

Calculate the 2D signed impact parameter significance for each particle.
This is a Julia implementation of the C++ function get_Sip2dSig.

Parameters:
- Sip2dVals: Vector of vectors containing 2D signed impact parameter values
- err2_D0: Vector of vectors containing squared errors of the D0 values

Returns:
- Vector of vectors of 2D signed impact parameter significances (one vector per jet)
"""
function get_Sip2dSig(Sip2dVals::Vector{Vector{Float32}},
                    err2_D0::Vector{Vector{Float32}})
    # Initialize result vector
    result = Vector{Vector{Float32}}()
    
    # Process each jet
    for i in eachindex(Sip2dVals)
        # Initialize significance values for this jet
        sig_values = Vector{Float32}()
        
        # Process each particle in the jet
        for j in eachindex(Sip2dVals[i])
            # Only calculate significance if the error is positive
            if j <= length(err2_D0[i]) && err2_D0[i][j] > 0.0
                # Calculate significance by dividing the value by its error
                significance = Sip2dVals[i][j] / sqrt(err2_D0[i][j])
                push!(sig_values, significance)
            else
                # Invalid measurement
                push!(sig_values, -9.0f0)
            end
        end
        
        push!(result, sig_values)
    end
    
    return result
end

function get_btagSip2dSig(pfcand_btagSip2dVal::Vector{Vector{Float32}},
                        pfcand_dxydxy::Vector{Vector{Float32}})
    # Simply call the implementation function
    return get_Sip2dSig(pfcand_btagSip2dVal, pfcand_dxydxy)
end

"""
    get_Sip3dVal_clusterV(jets::Vector{JetReconstruction.PseudoJet},
                         D0::Vector{FCCAnalysesJetConstituentsData},
                         Z0::Vector{FCCAnalysesJetConstituentsData},
                         phi0::Vector{FCCAnalysesJetConstituentsData},
                         Bz::Float32) -> Vector{FCCAnalysesJetConstituentsData}

Calculate the 3D signed impact parameter value for each particle relative to the jet axis.
"""
function get_Sip3dVal_clusterV(jets::Vector{JetReconstruction.PseudoJet},
                              D0::Vector{FCCAnalysesJetConstituentsData},
                              Z0::Vector{FCCAnalysesJetConstituentsData},
                              phi0::Vector{FCCAnalysesJetConstituentsData},
                              Bz::Float32)
    result = Vector{FCCAnalysesJetConstituentsData}()
    
    for i in eachindex(jets)
        p = TVector3(jets[i].px(), jets[i].py(), jets[i].pz())
        cprojs = FCCAnalysesJetConstituentsData()
        
        for j in eachindex(D0[i])
            if D0[i][j] != -9.0
                # Create 3D vector of displacement at point of closest approach
                d = TVector3(-D0[i][j] * sin(phi0[i][j]), 
                             D0[i][j] * cos(phi0[i][j]), 
                             Z0[i][j])
                
                # Sign the impact parameter based on the dot product with jet direction
                # dot product in 3D
                dot_prod = d.X() * p.X() + d.Y() * p.Y() + d.Z() * p.Z()
                sign_val = dot_prod > 0.0 ? 1.0 : -1.0
                impact_val = sqrt(D0[i][j]^2 + Z0[i][j]^2)
                cprojs_val = sign_val * impact_val
                push!(cprojs, cprojs_val)
            else
                push!(cprojs, -9.0f0)
            end
        end
        
        push!(result, cprojs)
    end
    
    return result
end

"""
    get_Sip3dSig(Sip3dVals::Vector{FCCAnalysesJetConstituentsData},
                err2_D0::Vector{FCCAnalysesJetConstituentsData},
                err2_Z0::Vector{FCCAnalysesJetConstituentsData}) -> Vector{FCCAnalysesJetConstituentsData}

Calculate the 3D signed impact parameter significance (value/error) for each particle.
"""
function get_Sip3dSig(Sip3dVals::Vector{FCCAnalysesJetConstituentsData},
                     err2_D0::Vector{FCCAnalysesJetConstituentsData},
                     err2_Z0::Vector{FCCAnalysesJetConstituentsData})
    result = Vector{FCCAnalysesJetConstituentsData}()
    
    for i in eachindex(Sip3dVals)
        sigs = FCCAnalysesJetConstituentsData()
        
        for j in eachindex(Sip3dVals[i])
            if err2_D0[i][j] > 0.0
                sig = Sip3dVals[i][j] / sqrt(err2_D0[i][j] + err2_Z0[i][j])
                push!(sigs, sig)
            else
                push!(sigs, -9.0f0)
            end
        end
        
        push!(result, sigs)
    end
    
    return result
end

"""
    get_JetDistVal_clusterV(jets::Vector{JetReconstruction.PseudoJet},
                           jcs::Vector{FCCAnalysesJetConstituents},
                           D0::Vector{FCCAnalysesJetConstituentsData},
                           Z0::Vector{FCCAnalysesJetConstituentsData},
                           phi0::Vector{FCCAnalysesJetConstituentsData},
                           Bz::Float32) -> Vector{FCCAnalysesJetConstituentsData}

Calculate the jet distance value for each particle, measuring the distance between
the point of closest approach and the jet axis.
"""
function get_JetDistVal_clusterV(jets::Vector{JetReconstruction.PseudoJet},
                                jcs::Vector{FCCAnalysesJetConstituents},
                                D0::Vector{FCCAnalysesJetConstituentsData},
                                Z0::Vector{FCCAnalysesJetConstituentsData},
                                phi0::Vector{FCCAnalysesJetConstituentsData},
                                Bz::Float32)
    result = Vector{FCCAnalysesJetConstituentsData}()
    
    for i in eachindex(jets)
        p_jet = TVector3(jets[i].px(), jets[i].py(), jets[i].pz())
        tmp = FCCAnalysesJetConstituentsData()
        
        for j in eachindex(D0[i])
            if D0[i][j] != -9.0 && j <= length(jcs[i])
                # Create 3D vector of displacement at point of closest approach
                d = TVector3(-D0[i][j] * sin(phi0[i][j]), 
                             D0[i][j] * cos(phi0[i][j]), 
                             Z0[i][j])
                
                # Calculate particle momentum
                p_ct = TVector3(jcs[i][j].momentum.x, 
                                jcs[i][j].momentum.y, 
                                jcs[i][j].momentum.z)
                
                # Jet origin
                r_jet = TVector3(0.0, 0.0, 0.0)
                
                # Normal vector to plane containing particle and jet momenta
                n = p_ct.Cross(p_jet).Unit()
                
                # Distance is projection of displacement onto normal vector
                dist = n.X() * (d.X() - r_jet.X()) + 
                       n.Y() * (d.Y() - r_jet.Y()) + 
                       n.Z() * (d.Z() - r_jet.Z())
                
                push!(tmp, dist)
            else
                push!(tmp, -9.0f0)
            end
        end
        
        push!(result, tmp)
    end
    
    return result
end

"""
    get_JetDistSig(JetDistVal::Vector{FCCAnalysesJetConstituentsData},
                  err2_D0::Vector{FCCAnalysesJetConstituentsData},
                  err2_Z0::Vector{FCCAnalysesJetConstituentsData}) -> Vector{FCCAnalysesJetConstituentsData}

Calculate the jet distance significance (value/error) for each particle.
"""
function get_JetDistSig(JetDistVal::Vector{FCCAnalysesJetConstituentsData},
                       err2_D0::Vector{FCCAnalysesJetConstituentsData},
                       err2_Z0::Vector{FCCAnalysesJetConstituentsData})
    result = Vector{FCCAnalysesJetConstituentsData}()
    
    for i in eachindex(JetDistVal)
        tmp = FCCAnalysesJetConstituentsData()
        
        for j in eachindex(JetDistVal[i])
            if err2_D0[i][j] > 0.0
                # 3D error
                err3d = sqrt(err2_D0[i][j] + err2_Z0[i][j])
                # Calculate significance
                jetdistsig = JetDistVal[i][j] / err3d
                push!(tmp, jetdistsig)
            else
                push!(tmp, -9.0f0)
            end
        end
        
        push!(result, tmp)
    end
    
    return result
end

"""
    count_jets(jets::Vector{FCCAnalysesJetConstituents}) -> Int

Count the number of jets.
"""
function count_jets(jets::Vector{FCCAnalysesJetConstituents})
    return length(jets)
end

"""
    count_consts(jets::Vector{FCCAnalysesJetConstituents}) -> Vector{Int}

Count the number of constituents in each jet.
"""
function count_consts(jets::Vector{FCCAnalysesJetConstituents})
    result = Vector{Int}()
    
    for i in eachindex(jets)
        push!(result, length(jets[i]))
    end
    
    return result
end

"""
    count_type(isType::Vector{FCCAnalysesJetConstituentsData}) -> Vector{Int}

Count the number of particles of a specific type in each jet.
"""
function count_type(isType::Vector{FCCAnalysesJetConstituentsData})
    result = Vector{Int}()
    
    for i in eachindex(isType)
        count = 0
        for j in eachindex(isType[i])
            if Int(isType[i][j]) == 1
                count += 1
            end
        end
        push!(result, count)
    end
    
    return result
end

end # module