# Tests for the ReconstructedParticle2Track module

include("common.jl")

using ReconstructedParticle
using ReconstructedParticle2Track
using EDM4hep
using LorentzVectorHEP
using Test
using LinearAlgebra

## TODO: Same LorentzVector change as the ReconstructedParticle module unittest.

# Create sample reconstructed particles for testing
function create_test_particles()
    particles = EDM4hep.ReconstructedParticle[]
    
    # Electron
    electron = EDM4hep.ReconstructedParticle()
    electron.PDG = 11
    electron.energy = 10.0
    electron.momentum = EDM4hep.Vector3f(5.0, 3.0, 8.0)
    electron.mass = 0.000511
    electron.charge = -1.0
    electron.tracks_begin = 0
    push!(particles, electron)
    
    # Muon
    muon = EDM4hep.ReconstructedParticle()
    muon.PDG = 13
    muon.energy = 20.0
    muon.momentum = EDM4hep.Vector3f(8.0, 5.0, 17.0)
    muon.mass = 0.105658
    muon.charge = -1.0
    muon.tracks_begin = 1
    push!(particles, muon)
    
    # Positron
    positron = EDM4hep.ReconstructedParticle()
    positron.PDG = -11
    positron.energy = 15.0
    positron.momentum = EDM4hep.Vector3f(7.0, 6.0, 12.0)
    positron.mass = 0.000511
    positron.charge = 1.0
    positron.tracks_begin = 2
    push!(particles, positron)
    
    # Neutral particle (no track)
    photon = EDM4hep.ReconstructedParticle()
    photon.PDG = 22
    photon.energy = 25.0
    photon.momentum = EDM4hep.Vector3f(10.0, 8.0, 22.0)
    photon.mass = 0.0
    photon.charge = 0.0
    photon.tracks_begin = -1
    push!(particles, photon)
    
    return particles
end

# Create sample tracks for testing
function create_test_tracks()
    tracks = EDM4hep.TrackState[]
    
    # Track for electron
    track1 = EDM4hep.TrackState()
    track1.D0 = 0.01
    track1.Z0 = 0.02
    track1.phi = 0.5
    track1.omega = 0.1
    track1.tanLambda = 1.5
    track1.covMatrix = zeros(Float32, 15)
    track1.covMatrix[1] = 0.001  # D0 variance
    track1.covMatrix[3] = 0.002  # phi variance
    track1.covMatrix[6] = 0.003  # omega variance
    track1.covMatrix[10] = 0.004  # Z0 variance
    track1.covMatrix[15] = 0.005  # tanLambda variance
    push!(tracks, track1)
    
    # Track for muon
    track2 = EDM4hep.TrackState()
    track2.D0 = 0.02
    track2.Z0 = 0.03
    track2.phi = 0.7
    track2.omega = 0.05
    track2.tanLambda = 2.0
    track2.covMatrix = zeros(Float32, 15)
    track2.covMatrix[1] = 0.002  # D0 variance
    track2.covMatrix[3] = 0.003  # phi variance
    track2.covMatrix[6] = 0.004  # omega variance
    track2.covMatrix[10] = 0.005  # Z0 variance
    track2.covMatrix[15] = 0.006  # tanLambda variance
    push!(tracks, track2)
    
    # Track for positron
    track3 = EDM4hep.TrackState()
    track3.D0 = 0.015
    track3.Z0 = 0.025
    track3.phi = 0.6
    track3.omega = -0.08
    track3.tanLambda = 1.8
    track3.covMatrix = zeros(Float32, 15)
    track3.covMatrix[1] = 0.0015  # D0 variance
    track3.covMatrix[3] = 0.0025  # phi variance
    track3.covMatrix[6] = 0.0035  # omega variance
    track3.covMatrix[10] = 0.0045  # Z0 variance
    track3.covMatrix[15] = 0.0055  # tanLambda variance
    push!(tracks, track3)
    
    return tracks
end

@testset "ReconstructedParticle2Track module tests" begin
    particles = create_test_particles()
    tracks = create_test_tracks()
    
    @testset "Basic track parameter getters" begin
        # Test D0 getter
        d0_vals = ReconstructedParticle2Track.getRP2TRK_D0(particles, tracks)
        @test length(d0_vals) == 4
        @test d0_vals[1] ≈ 0.01
        @test d0_vals[2] ≈ 0.02
        @test d0_vals[3] ≈ 0.015
        @test d0_vals[4] ≈ -9.0  # No track for photon
        
        # Test Z0 getter
        z0_vals = ReconstructedParticle2Track.getRP2TRK_Z0(particles, tracks)
        @test length(z0_vals) == 4
        @test z0_vals[1] ≈ 0.02
        @test z0_vals[2] ≈ 0.03
        @test z0_vals[3] ≈ 0.025
        @test z0_vals[4] ≈ -9.0  # No track for photon
        
        # Test phi getter
        phi_vals = ReconstructedParticle2Track.getRP2TRK_phi(particles, tracks)
        @test length(phi_vals) == 4
        @test phi_vals[1] ≈ 0.5
        @test phi_vals[2] ≈ 0.7
        @test phi_vals[3] ≈ 0.6
        @test phi_vals[4] ≈ -9.0  # No track for photon
        
        # Test omega getter
        omega_vals = ReconstructedParticle2Track.getRP2TRK_omega(particles, tracks)
        @test length(omega_vals) == 4
        @test omega_vals[1] ≈ 0.1
        @test omega_vals[2] ≈ 0.05
        @test omega_vals[3] ≈ -0.08
        @test omega_vals[4] ≈ -9.0  # No track for photon
        
        # Test tanLambda getter
        tanLambda_vals = ReconstructedParticle2Track.getRP2TRK_tanLambda(particles, tracks)
        @test length(tanLambda_vals) == 4
        @test tanLambda_vals[1] ≈ 1.5
        @test tanLambda_vals[2] ≈ 2.0
        @test tanLambda_vals[3] ≈ 1.8
        @test tanLambda_vals[4] ≈ -9.0  # No track for photon
    end
    
    @testset "Track covariance matrix getters" begin
        # Test D0 covariance getter
        d0_cov_vals = ReconstructedParticle2Track.getRP2TRK_D0_cov(particles, tracks)
        @test length(d0_cov_vals) == 4
        @test d0_cov_vals[1] ≈ 0.001
        @test d0_cov_vals[2] ≈ 0.002
        @test d0_cov_vals[3] ≈ 0.0015
        @test d0_cov_vals[4] ≈ -9.0  # No track for photon
        
        # Test Z0 covariance getter
        z0_cov_vals = ReconstructedParticle2Track.getRP2TRK_Z0_cov(particles, tracks)
        @test length(z0_cov_vals) == 4
        @test z0_cov_vals[1] ≈ 0.004
        @test z0_cov_vals[2] ≈ 0.005
        @test z0_cov_vals[3] ≈ 0.0045
        @test z0_cov_vals[4] ≈ -9.0  # No track for photon
        
        # Test phi covariance getter
        phi_cov_vals = ReconstructedParticle2Track.getRP2TRK_phi_cov(particles, tracks)
        @test length(phi_cov_vals) == 4
        @test phi_cov_vals[1] ≈ 0.002
        @test phi_cov_vals[2] ≈ 0.003
        @test phi_cov_vals[3] ≈ 0.0025
        @test phi_cov_vals[4] ≈ -9.0  # No track for photon
        
        # Test omega covariance getter
        omega_cov_vals = ReconstructedParticle2Track.getRP2TRK_omega_cov(particles, tracks)
        @test length(omega_cov_vals) == 4
        @test omega_cov_vals[1] ≈ 0.003
        @test omega_cov_vals[2] ≈ 0.004
        @test omega_cov_vals[3] ≈ 0.0035
        @test omega_cov_vals[4] ≈ -9.0  # No track for photon
        
        # Test tanLambda covariance getter
        tanLambda_cov_vals = ReconstructedParticle2Track.getRP2TRK_tanLambda_cov(particles, tracks)
        @test length(tanLambda_cov_vals) == 4
        @test tanLambda_cov_vals[1] ≈ 0.005
        @test tanLambda_cov_vals[2] ≈ 0.006
        @test tanLambda_cov_vals[3] ≈ 0.0055
        @test tanLambda_cov_vals[4] ≈ -9.0  # No track for photon
    end
    
    @testset "Track selection and filtering" begin
        # Test getting track states
        track_states = ReconstructedParticle2Track.getRP2TRK(particles, tracks)
        @test length(track_states) == 3  # Electron, muon, positron (photon has no track)
        @test track_states[1].D0 ≈ 0.01
        @test track_states[2].D0 ≈ 0.02
        @test track_states[3].D0 ≈ 0.015
        
        # Test getting reconstructed indices with tracks
        track_indices = ReconstructedParticle2Track.get_recoindTRK(particles, tracks)
        @test length(track_indices) == 3
        @test track_indices == [1, 2, 3]
        
        # Test has track function
        has_track = ReconstructedParticle2Track.hasTRK(particles)
        @test length(has_track) == 4
        @test has_track[1] == true
        @test has_track[2] == true
        @test has_track[3] == true
        @test has_track[4] == false
        
        # Test track count function
        track_count = ReconstructedParticle2Track.getTK_n(tracks)
        @test track_count == 3
    end
    
    @testset "Magnetic field calculation" begin
        # Test Bz function for single particle
        bz_value = ReconstructedParticle2Track.Bz(particles, tracks)
        # Value depends on track parameters and constants, so we just check it's not -9
        @test bz_value != -9.0
        
        # Test Bz function for all particles
        bz_values = ReconstructedParticle2Track.getRP2TRK_Bz(particles, tracks)
        @test length(bz_values) == 4
        # First three should have real values, photon should have default value
        @test bz_values[1] != -9.0
        @test bz_values[2] != -9.0
        @test bz_values[3] != -9.0
        @test bz_values[4] ≈ -9.0
        
        # Check signs of Bz match charge signs
        @test sign(bz_values[1]) == sign(particles[1].charge)
        @test sign(bz_values[2]) == sign(particles[2].charge)
        @test sign(bz_values[3]) == sign(particles[3].charge)
    end
    
    @testset "Impact parameter calculations" begin
        # Create a test vertex at (0,0,0)
        vertex = TLorentzVector(0.0, 0.0, 0.0, 0.0)
        # Use a constant Bz field for testing
        bz_field = 2.0
        
        # Test dxy calculation
        dxy_values = ReconstructedParticle2Track.XPtoPar_dxy(particles, tracks, vertex, bz_field)
        @test length(dxy_values) == 4
        # We can't easily predict exact values but we can check they're reasonable
        @test dxy_values[1] != -9.0
        @test dxy_values[2] != -9.0
        @test dxy_values[3] != -9.0
        @test dxy_values[4] ≈ -9.0  # No track for photon
        
        # Test dz calculation
        dz_values = ReconstructedParticle2Track.XPtoPar_dz(particles, tracks, vertex, bz_field)
        @test length(dz_values) == 4
        @test dz_values[1] != -9.0
        @test dz_values[2] != -9.0
        @test dz_values[3] != -9.0
        @test dz_values[4] ≈ -9.0  # No track for photon
        
        # Test phi calculation
        phi_values = ReconstructedParticle2Track.XPtoPar_phi(particles, tracks, vertex, bz_field)
        @test length(phi_values) == 4
        @test phi_values[1] != -9.0
        @test phi_values[2] != -9.0
        @test phi_values[3] != -9.0
        @test phi_values[4] ≈ -9.0  # No track for photon
        
        # Test curvature calculation
        c_values = ReconstructedParticle2Track.XPtoPar_C(particles, tracks, bz_field)
        @test length(c_values) == 4
        @test c_values[1] != -9.0
        @test c_values[2] != -9.0
        @test c_values[3] != -9.0
        @test c_values[4] ≈ -9.0  # No track for photon
        
        # Test cot(theta) calculation
        ct_values = ReconstructedParticle2Track.XPtoPar_ct(particles, tracks, bz_field)
        @test length(ct_values) == 4
        @test ct_values[1] != -9.0
        @test ct_values[2] != -9.0
        @test ct_values[3] != -9.0
        @test ct_values[4] ≈ -9.0  # No track for photon
    end
end