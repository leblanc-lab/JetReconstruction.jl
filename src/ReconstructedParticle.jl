module ReconstructedParticle

using JetReconstruction
using LorentzVectorHEP
using EDM4hep
using StaticArrays
using LinearAlgebra
using StructArrays: StructVector

const JetConstituents = StructVector{EDM4hep.ReconstructedParticle}

"""
    get_p(inputs::JetConstituents) -> Vector{Float64}

Get the momentum magnitude of each particle.
"""
function get_p(inputs::JetConstituents)
    result = Float64[]
    for p in inputs 
        momentum_mag = sqrt(p.momentum.x^2 + p.momentum.y^2 + p.momentum.z^2)
        push!(result, momentum_mag)
    end
    return result
end

"""
    get_px(inputs::JetConstituents) -> Vector{Float64}

Get the x-component of momentum for each particle.
"""
function get_px(inputs::JetConstituents)
    result = Float64[]
    for p in inputs
        push!(result, p.momentum.x)
    end
    return result
end

"""
    get_py(inputs::JetConstituents) -> Vector{Float64}

Get the y-component of momentum for each particle.
"""
function get_py(inputs::JetConstituents)
    result = Float64[]
    for p in inputs
        push!(result, p.momentum.y)
    end
    return result
end

"""
    get_pz(inputs::JetConstituents) -> Vector{Float64}

Get the z-component of momentum for each particle.
"""
function get_pz(inputs::JetConstituents)
    result = Float64[]
    for p in inputs
        push!(result, p.momentum.z)
    end
    return result
end

"""
    get_pt(inputs::JetConstituents) -> Vector{Float64}

Get the transverse momentum of each particle.
"""
function get_pt(inputs::JetConstituents)
    result = Float64[]
    for p in inputs
        pt_val = sqrt(p.momentum.x^2 + p.momentum.y^2)
        push!(result, pt_val)
    end
    return result
end

"""
    to_lorentz_vector(p::EDM4hep.ReconstructedParticle) -> LorentzVector{Float64}

Convert a ReconstructedParticle to a LorentzVector.
"""
function to_lorentz_vector(p::EDM4hep.ReconstructedParticle)
    return LorentzVector(Float64(p.energy), 
                         Float64(p.momentum.x), 
                         Float64(p.momentum.y), 
                         Float64(p.momentum.z))
end

"""
    get_eta(inputs::JetConstituents) -> Vector{Float64}

Get the pseudorapidity of each particle.
"""
function get_eta(inputs::JetConstituents)
    result = Float64[]
    for p in inputs
        lv = to_lorentz_vector(p)
        push!(result, eta(lv))
    end
    return result
end

"""
    get_theta(inputs::JetConstituents) -> Vector{Float64}

Get the polar angle of each particle.
"""
function get_theta(inputs::JetConstituents)
    result = Float64[]
    for p in inputs
        # Calculate theta from momentum components
        p_mag = sqrt(p.momentum.x^2 + p.momentum.y^2 + p.momentum.z^2)
        if p_mag > 0
            theta_val = acos(p.momentum.z / p_mag)
        else
            theta_val = 0.0
        end
        push!(result, theta_val)
    end
    return result
end

"""
    get_phi(inputs::JetConstituents) -> Vector{Float64}

Get the azimuthal angle of each particle.
"""
function get_phi(inputs::JetConstituents)
    result = Float64[]
    for p in inputs
        lv = to_lorentz_vector(p)
        push!(result, phi(lv))
    end
    return result
end

"""
    get_rapidity(inputs::JetConstituents) -> Vector{Float64}

Get the rapidity of each particle.
"""
function get_rapidity(inputs::JetConstituents)
    result = Float64[]
    for p in inputs
        lv = to_lorentz_vector(p)
        push!(result, rapidity(lv))
    end
    return result
end

"""
    get_energy(inputs::JetConstituents) -> Vector{Float64}

Get the energy of each particle.
"""
function get_energy(inputs::JetConstituents)
    result = Float64[]
    for p in inputs
        push!(result, p.energy)
    end
    return result
end

"""
    get_mass(inputs::JetConstituents) -> Vector{Float64}

Get the mass of each particle.
"""
function get_mass(inputs::JetConstituents)
    result = Float64[]
    for p in inputs
        push!(result, p.mass)
    end
    return result
end

"""
    get_charge(inputs::JetConstituents) -> Vector{Float64}

Get the charge of each particle.
"""
function get_charge(inputs::JetConstituents)
    result = Float64[]
    for p in inputs
        push!(result, p.charge)
    end
    return result
end

"""
    get_type(inputs::JetConstituents) -> Vector{Int32}

Get the PDG type of each particle.
"""
function get_type(inputs::JetConstituents)
    result = Int32[]
    for p in inputs
        push!(result, p.PDG)
    end
    return result
end

"""
    get_lorentz_vector(inputs::JetConstituents) -> Vector{LorentzVector{Float64}}

Convert each particle to a LorentzVector.
"""
function get_lorentz_vector(inputs::JetConstituents)
    result = LorentzVector{Float64}[]
    for p in inputs
        push!(result, to_lorentz_vector(p))
    end
    return result
end

"""
    get_tlv(inputs::JetConstituents) -> Vector{LorentzVector{Float64}}

Alias for get_lorentz_vector.
"""
const get_tlv = get_lorentz_vector

"""
    get_tlv(input::EDM4hep.ReconstructedParticle) -> LorentzVector{Float64}

Convert a single particle to a LorentzVector.
"""
function get_tlv(input::EDM4hep.ReconstructedParticle)
    return to_lorentz_vector(input)
end

"""
    get_n(inputs::JetConstituents) -> Int

Get the number of particles.
"""
function get_n(inputs::JetConstituents)
    return length(inputs)
end

"""
    get_P4vis(inputs::JetConstituents) -> LorentzVector{Float64}

Get the visible 4-momentum by summing all particles.
"""
function get_P4vis(inputs::JetConstituents)
    result = LorentzVector(0.0, 0.0, 0.0, 0.0)
    for p in inputs
        result += to_lorentz_vector(p)
    end
    return result
end

"""
    merge(x::JetConstituents, y::JetConstituents) -> JetConstituents

Merge two collections of particles.
"""
function Base.merge(x::JetConstituents, y::JetConstituents)
    result = JetConstituents()
    append!(result, x)
    append!(result, y)
    return result
end

"""
    remove(x::JetConstituents, y::JetConstituents) -> JetConstituents

Remove particles in y from x.
"""
function remove(x::JetConstituents, y::JetConstituents)
    result = copy(x)
    for p in y
        # Find and remove the matching particle
        idx = findfirst(particle -> 
            isapprox(particle.mass, p.mass, atol=1e-8) && 
            isapprox(particle.momentum.x, p.momentum.x, atol=1e-8) && 
            isapprox(particle.momentum.y, p.momentum.y, atol=1e-8) && 
            isapprox(particle.momentum.z, p.momentum.z, atol=1e-8), 
            result)
        
        if idx !== nothing
            deleteat!(result, idx)
        end
    end
    return result
end

"""
    sel_pt(min_pt::Float64) -> Function

Select particles with transverse momentum greater than min_pt.
"""
function sel_pt(min_pt::Float64)
    return function(inputs::JetConstituents)
        result = JetConstituents()
        for p in inputs
            pt_val = sqrt(p.momentum.x^2 + p.momentum.y^2)
            if pt_val > min_pt
                push!(result, p)
            end
        end
        return result
    end
end

"""
    sel_eta(max_eta::Float64) -> Function

Select particles with absolute pseudorapidity less than max_eta.
"""
function sel_eta(max_eta::Float64)
    return function(inputs::JetConstituents)
        result = JetConstituents()
        for p in inputs
            lv = to_lorentz_vector(p)
            if abs(eta(lv)) < max_eta
                push!(result, p)
            end
        end
        return result
    end
end

"""
    sel_p(min_p::Float64, max_p::Float64=1e10) -> Function

Select particles with momentum between min_p and max_p.
Returns a JetConstituents (StructVector) object.
"""
function sel_p(min_p::Float64=0.0, max_p::Float64=1e10)
    return function(inputs::JetConstituents)
        # Collect indices of particles that pass the momentum cut
        selected_indices = Int[]
        
        for (i, p) in enumerate(inputs)
            p_mag = sqrt(p.momentum.x^2 + p.momentum.y^2 + p.momentum.z^2)
            if min_p < p_mag < max_p
                push!(selected_indices, i)
            end
        end
        
        # Return a subset of the original StructVector
        if !isempty(selected_indices)
            return inputs[selected_indices]
        else
            # Create an empty JetConstituents with the same structure
            return StructVector{EDM4hep.ReconstructedParticle}(similar.(fieldarrays(inputs), 0))
            # return StructVector{EDM4hep.ReconstructedParticle}(inputs,0)
        end
    end
end

# function sel_p(min_p::Float64, max_p::Float64=1e10)
#     return function(inputs::JetConstituents)
#         result = JetConstituents()
#         for p in inputs
#             p_mag = sqrt(p.momentum.x^2 + p.momentum.y^2 + p.momentum.z^2)
#             if min_p < p_mag < max_p
#                 push!(result, p)
#             end
#         end
#         return result
#     end
# end

"""
    sel_charge(charge::Int, abs_charge::Bool=false) -> Function

Select particles by charge.
"""
function sel_charge(charge::Int, abs_charge::Bool=false)
    return function(inputs::JetConstituents)
        result = JetConstituents()
        for p in inputs
            if abs_charge
                if abs(round(Int, p.charge)) == charge
                    push!(result, p)
                end
            else
                if round(Int, p.charge) == charge
                    push!(result, p)
                end
            end
        end
        return result
    end
end

"""
    sel_type(type::Int) -> Function

Select particles by PDG type.
"""
function sel_type(type::Int)
    return function(inputs::JetConstituents)
        result = JetConstituents()
        for p in inputs
            if p.PDG == type
                push!(result, p)
            end
        end
        return result
    end
end

"""
    sel_absType(type::Int) -> Function

Select particles by absolute PDG type.
"""
function sel_absType(type::Int)
    if type < 0
        error("sel_absType: Type must be non-negative")
    end
    
    return function(inputs::JetConstituents)
        result = JetConstituents()
        for p in inputs
            if abs(p.PDG) == type
                push!(result, p)
            end
        end
        return result
    end
end

function sel_tag()
    ### TODO: Implement this function
end

"""
    get(indices::StructVector{ObjectID}, particles::JetConstituents) -> JetConstituents

Get reconstructed particles referenced by ObjectIDs.
"""
function get(indices::StructVector{ObjectID}, particles::JetConstituents)
    # Collect valid indices from ObjectIDs
    valid_indices = Int[]
    
    for i in indices
        index = i.index
        if index > 0 && index <= length(particles)
            push!(valid_indices, index)
        end
    end
    
    # If we have valid indices, create a subset of the original StructVector
    if !isempty(valid_indices)
        return particles[valid_indices]
    else
        # Create an empty JetConstituents with the same structure
        return StructVector{EDM4hep.ReconstructedParticle}(similar.(fieldarrays(particles), 0))
    end
end

end # module
