using JetReconstruction

#do I need to account for different versions of fastjet? I assumed no 
#two types can't inherit one another if I need to instantate??? 

mutable struct SoftKiller <: TilingBase
    _ymax::Float64
    _ymin::Float64
    _requested_drap::Float64
    _requested_dphi::Float64
    #selector 
    _ntotal::Int64
    _ngood::Int64
    _dy::Float64
    _dphi::Float64
    _cell_area::Float64
    _inverse_dy::Float64
    _inverse_dphi::Float64
    _ny::Int64
    _nphi::Int64

    function SoftKiller(rapmin::Float64, rapmax::Float64, drap::Float64, dphi::Float64)
        print("4 variables \n")
        grid = new(rapmax, rapmin, drap, dphi, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
        _setup_grid(grid)
        print(description(grid))
        grid 
    end

    function SoftKiller(rapmax::Float64, grid_size::Float64)
        print("2 variables\n")
        grid = new(rapmax, -rapmax, grid_size, grid_size, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
        _setup_grid(grid)
        print(description(grid))
        grid 
    end

end 

tile_index(sk:: SoftKiller, p::PseudoJet)::Int64 = begin 
    y_minus_ymin = rapidity(p)- sk._ymin
    if y_minus_ymin < 0 
        return -1   
    end 
    
    iy = round(Int64,y_minus_ymin * sk._inverse_dy)
    if iy >= sk._ny
        return -1 
    end 

    iphi = round(Int64,phi(p)*sk._inverse_dphi)
    if iphi == sk._nphi
        iphi =0
    end 
   
    res = round(Int64,  iy*sk._nphi + iphi) 

    res + 1
end 


_setup_grid(sk:: SoftKiller) = begin 
    @assert sk._ymax > sk._ymin
    @assert sk._requested_drap > 0
    @assert sk._requested_dphi > 0 

    ny_double = (sk._ymax-sk._ymin) / sk._requested_drap
    sk._ny = max(round(Int64, ny_double+0.5),1)
    sk._dy = (sk._ymax-sk._ymin) / sk._ny
    sk._inverse_dy = sk._ny/(sk._ymax-sk._ymin)

    sk._nphi = round(Int64,(2* π) / sk._requested_dphi + 0.5)
    sk._dphi = (2* π) / sk._nphi
    sk._inverse_dphi = sk._nphi/(2*π)

    @assert sk._ny >=1 and sk._nphi >=1 

    sk._ntotal = sk._nphi * sk._ny
    sk._cell_area = sk._dy * sk._dphi

end 

description(sk::SoftKiller)::String = begin 
    #from definiton of is_initialised  in RectangularGrid.hh
    if sk._ntotal <= 0
        return "Uninitialised rectangular grid" 
    end 

    descr = "rectangular grid with rapidity extent $(sk._ymin) < rap < $(sk._ymax) \n total tiles  $(sk._ntotal) \n "
    descr *= "tile size drap x dphi = $(sk._dy) x $(sk._dphi)"

    #Selector implementation desn't exist 
    descr 
end 

n_tiles(sk:: SoftKiller)::Int64 = begin 
    sk._ntotal
end 

n_good_tiles(sk:: SoftKiller)::Int64 = begin 
    #since the selector file is not implemented this for now will return sk._ntotal
    #to pass the aaertion in apply, when implemented it should return this  
    #sk._ngood
    sk._ntotal
end 

tile_is_good(sk:: SoftKiller,itile::Int64)::Bool = begin #requires selector
    true
end 

tile_area(sk:: SoftKiller,itile::Int64)::Float64 = begin
    sk.mean_tile_area()
end

mean_tile_area(sk:: SoftKiller)::Float64 = begin 
    sk._dphi*sk._dy
end 


is_initialized(sk:: SoftKiller)::Bool = begin
    sk._ntotal > 0
end 

select_ABS_RAP_max(event, absrapmax) = begin
    filtered_events = filter(e -> begin
        abs(rapidity(e)) <= absrapmax
    end, event)
    return filtered_events
end

apply(sk::SoftKiller, event::Vector{PseudoJet}, reduced_event::Vector{PseudoJet}, pt_threshold::Float64) = begin 

    if (n_tiles(sk) < 2)
        throw("SoftKiller not properly initialised.")
    end 


    #@assert event!=reduced_event #-> compares addresses - need to check how it is done with Julia 
    @assert all_tiles_equal_area() #-> can't get accessed with sk. all_tiles_equal_area() 

    #fills the lector of length n_tiles with 0's
    max_pt2 = fill(0.0, n_tiles(sk))

    #starts from 1 not 0! 
    for ev in event
        if (ev == isnothing)
            continue
        end 
        index = tile_index(sk,ev)
        if (index < 0)
            continue
        end 
        max_pt2[index] = max(max_pt2[index],pt2(ev))

       
    end 

    #no here is this for loop that handles the case when 
    #good tiles and tiles are not equal but I assume a selector wi used then 
    #since that's how good tiles are determined 

    sort!(max_pt2)

    int_median_pos = length(max_pt2) ÷ 2
    pt2cut = (1+1e-12)*max_pt2[int_median_pos]

    indices = Int64[]
    for (i, ps_jet) in enumerate(event)
        if  ps_jet === nothing || pt2(ps_jet) >= pt2cut
            push!(indices, i)
        end
    end

    resize!(reduced_event, length(indices)) 
    for (i, idx) in enumerate(indices)
        reduced_event[i] = event[idx]
    end

    pt_threshold = sqrt(pt2cut);

    return reduced_event, pt_threshold

end 


plot_set_up(Y::Vector{Float64}, Phi::Vector{Float64}, pt::Vector{Float64}, color::Vector{String},  plot_title::String) = begin
    y_min, y_max = -5.0, 5.0
    
    phi_min, phi_max = round(minimum(Phi)), maximum(Phi)

    x = y_min:0.4:y_max
    y = phi_min:0.4:phi_max

    min_pt, max_pt = minimum(pt), maximum(pt)
    marker_sizes = 3 .+ 16 .*((pt .- min_pt)./(max_pt - min_pt))

    format(lines) = begin 
        return [isinteger(line) ? string(line) : "" for line in lines]
    end

    x_labels = format(x)
    y_labels = format(y)

    p = scatter(Y, Phi,
        xlabel="Rapidity (y)",
        ylabel="Azimuthal Angle (φ)",
        title=plot_title,
        markersize=marker_sizes, 
        xticks=(x, x_labels),  
        yticks=(y, y_labels),  
        xlims=(y_min, y_max), 
        ylims=(phi_min, phi_max),  
        grid=true,  
        framestyle=:box,
        color=color;
        alpha=0.6,
        legend=false  
    )

    savefig(p, plot_title * ".png")
end
