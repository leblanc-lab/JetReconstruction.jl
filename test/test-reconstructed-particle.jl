# Tests for the ReconstructedParticle module

include("common.jl")

using ReconstructedParticle
using EDM4hep

## TODO: I think I can change the Vector3f to a LorentzVector from the LorentzHEP. 
## The Test in FCCAnalyses is using TLorentzVector, so I should be able to use it here too.

# Create sample particles for testing
function create_test_particles()
    particles = EDM4hep.ReconstructedParticle[]
    
    # Electron
    electron = EDM4hep.ReconstructedParticle()
    electron.PDG = 11
    electron.energy = 10.0
    electron.momentum = EDM4hep.Vector3f(5.0, 3.0, 8.0)
    electron.mass = 0.000511
    electron.charge = -1.0
    push!(particles, electron)
    
    # Muon
    muon = EDM4hep.ReconstructedParticle()
    muon.PDG = 13
    muon.energy = 20.0
    muon.momentum = EDM4hep.Vector3f(8.0, 5.0, 17.0)
    muon.mass = 0.105658
    muon.charge = -1.0
    push!(particles, muon)
    
    # Positron
    positron = EDM4hep.ReconstructedParticle()
    positron.PDG = -11
    positron.energy = 15.0
    positron.momentum = EDM4hep.Vector3f(7.0, 6.0, 12.0)
    positron.mass = 0.000511
    positron.charge = 1.0
    push!(particles, positron)
    
    # Pion
    pion = EDM4hep.ReconstructedParticle()
    pion.PDG = 211
    pion.energy = 25.0
    pion.momentum = EDM4hep.Vector3f(10.0, 8.0, 22.0)
    pion.mass = 0.13957
    pion.charge = 1.0
    push!(particles, pion)
    
    return particles
end

@testset "ReconstructedParticle module tests" begin
    particles = create_test_particles()
    
    @testset "Basic property getters" begin
        # Test energy getter
        energies = ReconstructedParticle.get_energy(particles)
        @test length(energies) == 4
        @test energies[1] ≈ 10.0
        @test energies[2] ≈ 20.0
        @test energies[3] ≈ 15.0
        @test energies[4] ≈ 25.0
        
        # Test momentum getters
        px_vals = ReconstructedParticle.get_px(particles)
        py_vals = ReconstructedParticle.get_py(particles)
        pz_vals = ReconstructedParticle.get_pz(particles)
        
        @test px_vals[1] ≈ 5.0
        @test py_vals[1] ≈ 3.0
        @test pz_vals[1] ≈ 8.0
        
        # Test pt getter
        pt_vals = ReconstructedParticle.get_pt(particles)
        @test pt_vals[1] ≈ sqrt(5.0^2 + 3.0^2)
        @test pt_vals[2] ≈ sqrt(8.0^2 + 5.0^2)
        
        # Test mass getter
        mass_vals = ReconstructedParticle.get_mass(particles)
        @test mass_vals[1] ≈ 0.000511
        @test mass_vals[2] ≈ 0.105658
        @test mass_vals[3] ≈ 0.000511
        @test mass_vals[4] ≈ 0.13957
        
        # Test charge getter
        charge_vals = ReconstructedParticle.get_charge(particles)
        @test charge_vals[1] ≈ -1.0
        @test charge_vals[2] ≈ -1.0
        @test charge_vals[3] ≈ 1.0
        @test charge_vals[4] ≈ 1.0
        
        # Test type getter
        type_vals = ReconstructedParticle.get_type(particles)
        @test type_vals[1] == 11
        @test type_vals[2] == 13
        @test type_vals[3] == -11
        @test type_vals[4] == 211
        
        # Test n getter
        @test ReconstructedParticle.get_n(particles) == 4
    end
    
    @testset "LorentzVector conversion" begin
        # Test lorentz vector getter for all particles
        lvs = ReconstructedParticle.get_tlv(particles)
        @test length(lvs) == 4
        
        # Test individual lorentz vector
        lv = ReconstructedParticle.get_tlv(particles[1])
        @test energy(lv) ≈ 10.0
        @test px(lv) ≈ 5.0
        @test py(lv) ≈ 3.0
        @test pz(lv) ≈ 8.0
        
        # Test visible 4-momentum
        p4vis = ReconstructedParticle.get_P4vis(particles)
        @test energy(p4vis) ≈ 10.0 + 20.0 + 15.0 + 25.0
        @test px(p4vis) ≈ 5.0 + 8.0 + 7.0 + 10.0
        @test py(p4vis) ≈ 3.0 + 5.0 + 6.0 + 8.0
        @test pz(p4vis) ≈ 8.0 + 17.0 + 12.0 + 22.0
    end
    
    @testset "Particle selection functions" begin
        # Test pt selection
        pt_selector = ReconstructedParticle.sel_pt(7.0)
        high_pt_particles = pt_selector(particles)
        @test length(high_pt_particles) == 3  # muon, positron, pion
        @test high_pt_particles[1].PDG == 13
        @test high_pt_particles[2].PDG == -11
        @test high_pt_particles[3].PDG == 211
        
        # Test eta selection
        eta_selector = ReconstructedParticle.sel_eta(1.0)
        central_particles = eta_selector(particles)
        # Calculate expected etas
        particle_lvs = [ReconstructedParticle.get_tlv(p) for p in particles]
        particle_etas = [eta(lv) for lv in particle_lvs]
        expected_count = count(abs.(particle_etas) .< 1.0)
        @test length(central_particles) == expected_count
        
        # Test type selection
        type_selector = ReconstructedParticle.sel_type(11)
        electrons = type_selector(particles)
        @test length(electrons) == 1
        @test electrons[1].PDG == 11
        
        # Test abs type selection
        abs_type_selector = ReconstructedParticle.sel_absType(11)
        leptons = abs_type_selector(particles)
        @test length(leptons) == 2
        @test leptons[1].PDG == 11
        @test leptons[2].PDG == -11
        
        # Test charge selection
        neg_charge_selector = ReconstructedParticle.sel_charge(-1, false)
        neg_particles = neg_charge_selector(particles)
        @test length(neg_particles) == 2
        @test neg_particles[1].PDG == 11
        @test neg_particles[2].PDG == 13
        
        # Test abs charge selection
        unit_charge_selector = ReconstructedParticle.sel_charge(1, true)
        unit_charge_particles = unit_charge_selector(particles)
        @test length(unit_charge_particles) == 4  # all have |charge| = 1
    end
    
    @testset "Collection manipulation" begin
        particles1 = particles[1:2]  # electron, muon
        particles2 = particles[3:4]  # positron, pion
        
        # Test merge
        merged = Base.merge(particles1, particles2)
        @test length(merged) == 4
        @test merged[1].PDG == 11
        @test merged[3].PDG == -11
        
        # Test remove
        removed = ReconstructedParticle.remove(particles, particles1)
        @test length(removed) == 2
        @test removed[1].PDG == -11
        @test removed[2].PDG == 211
    end
end