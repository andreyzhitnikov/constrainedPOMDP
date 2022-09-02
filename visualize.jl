
gr()
function circleShape(h,k,r)
    # draw circle
    th = LinRange(0, 2*π,500)
    h .+ r*sin.(th), k .+ r*cos.(th)
end

function Brightness(pomdp, x, y)
    scale = 1.
    brightness = 1
    for (id,_) in enumerate( pomdp.beacons[:,1])
        if norm([x,y] - pomdp.beacons[id,:],2) < pomdp.d
            brightness = ((x - pomdp.beacons[id,1])^2 + (y - pomdp.beacons[id,2])^2)* scale
        end
    end
    return brightness
end 

function plotBeacon(plt, pomdp)
    inv_grays = cgrad([RGB(1.0,1.0,1.0),RGB(0.2,0.2,0.2)])
    x = range(-3, 15, length = 300)

    plt = heatmap!(x, x, (x...) -> Brightness(pomdp, x[1], x[2]), c = inv_grays,
    #bg_inside=:black,
    linewidth=0,
    clims = (0, 0.8),
    fill=true,
    cbar=false,
    #axis=true,
    border=:none,
    aspect_ratio = 1,
    background_color_inside = RGB(0.2,0.2,0.2),
    background_color = RGB(0.2,0.2,0.2),
    #xlims =(xsmin, xsmax),
    #ylims =(ysmin, ysmax),
    seriescolor = inv_grays,
    seriestype = :heatmap,
    label = "beacon")

end

function visualizeTrajectory(planner::Planner, x_gt, b_gt, b_PF; obstacles_title="low reward area", alg_name="noname", iter=0)
    Plots.theme(:dark)
    plot()
    Obs_trajectory = scatter!(planner.pomdp.beacons[:], markersize = 25, markercolor = :transparent, 
    markerstrokecolor ="blue", markerstrokewidth=1, label="")
    plotBeacon(Obs_trajectory, planner.pomdp)
  #Obs_trajectory = scatter!([x_gt[1][1]],[x_gt[1][2]],color="red", label="start",legend=:topleft)
    #Obs_trajectory = scatter!([x_gt[end][1]], [x_gt[end][2]],color="green",label="final pose",legend=:topleft)
    #Obs_trajectory = scatter!([x[1] for x in o_gt], [x[2] for x in o_gt],color="yellow", label="o_gt",legend=:topleft)
    #Obs_trajectory = scatter!([planner.pomdp.beacons[1]], [planner.pomdp.beacons[2]],color="red", label="beacons", markershape=:utriangle)
    
    #covellipse!(b_gt[1].μ, b_gt[1].Σ, showaxes=true, n_std=2,color="cyan",label="belief",legend=:topleft)
    #for i in 2:length(b_gt)
    #    covellipse!(b_gt[i].μ, b_gt[i].Σ, showaxes=true, n_std=2,color="cyan",label="")
    #end
    #Σ0 = initBelief().Σ
    #covellipse!(planner.pomdp.beacons[1,:], planner.pomdp.d*Σ0, showaxes=false, n_std=1,color="white",label="")
    #for i in 2:length(planner.pomdp.beacons[:,1])
    #    covellipse!(planner.pomdp.beacons[i,:], planner.pomdp.d*Σ0, showaxes=false, n_std=1,color="white",label="")
    #end
    
    var_trajectory = []
    color_range = [( 255, 255, 255), ( 39,  64, 139)]
    for i in 1:length(b_PF)
        clr = color_range[1] .+ i.*(color_range[2].-color_range[1])./length(b_PF)
        clr = clr ./ 255
        if i == 5
            lbl = "" #"particles"
        else
            lbl = ""
        end
        Particle_trajectory = scatter!([p[1] for p in particles(b_PF[i])], [p[2] for p in particles(b_PF[i])], color=RGB(clr...), markersize=3, label=lbl,markerstrokewidth=0)
        #=Particle_trajectory_x = [p[1] for p in particles(b_PF[i])]
        Particle_trajectory_y = [p[2] for p in particles(b_PF[i])]
        var_x = var(Particle_trajectory_x)
        var_y = var(Particle_trajectory_y)
        cov_xy = cov(Particle_trajectory_x,Particle_trajectory_y)
        cov_yx = cov(Particle_trajectory_y,Particle_trajectory_x)
        Σ = [var_x cov_xy; cov_yx var_y] + 10^-8 * [1 0; 0 1]
        μ = [mean(Particle_trajectory_x), mean(Particle_trajectory_y)]
        p = MvNormal(μ, Σ)
        f(x,y) = pdf(p, [x,y])
        X = range(-3, 13, length=1000)
        Y = range(-3, 13, length=1000)
        Obs_trajectory = contourf!(X, Y, f, color=:viridis)=#
    end

    # obstacles
    for i in 1:length(planner.pomdp.obstacles[:,1])
        # circleShape(h,k,r) - (h,k) denote center of the circle, r radius
        if i == 1
            lbl = obstacles_title
        else
            lbl = ""
        end
        Obs_trajectory = plot!(circleShape(planner.pomdp.obstacles[i,1], planner.pomdp.obstacles[i,2], planner.pomdp.obsRadii), seriestype = [:shape,], lw = 0.1, 
                                            c = :red, linecolor = :black, legend = false, fillalpha = 0.4, aspect_ratio = 1, label=lbl)
    end

    # goal
    #Obs_trajectory = plot!(circleShape(planner.pomdp.goal[1], planner.pomdp.goal[2], planner.pomdp.goalRadii), seriestype = [:shape,], lw = 0.1, 
    #c = :green, linecolor = :black, legend = false, fillalpha = 0.8, aspect_ratio = 1)
    Obs_trajectory = scatter!([planner.pomdp.goal[1]], [planner.pomdp.goal[2]], lw = 0.1, 
    markershape=:star5, markersize = 30,
    c = :green, linecolor = :black, legend = true, fillalpha = 0.8, aspect_ratio = 1,label="goal")

    # init_pose
    #Obs_trajectory = plot!(circleShape(planner.pomdp.goal[1], planner.pomdp.goal[2], planner.pomdp.goalRadii), seriestype = [:shape,], lw = 0.1, 
    #c = :green, linecolor = :black, legend = false, fillalpha = 0.8, aspect_ratio = 1)
    Obs_trajectory = scatter!([x_gt[1][1]], [x_gt[1][2]], lw = 4, 
    markershape=:cross, markersize = 30,
    c = :yellow, linecolor = :black, legend = true, fillalpha = 1, aspect_ratio = 1,label="initial pose")

    Obs_trajectory = scatter!([x[1] for x in x_gt], [x[2] for x in x_gt],color="yellow", label="ground truth",legend=:topleft)
    Obs_trajectory = plot!([x[1] for x in x_gt], [x[2] for x in x_gt],color="yellow", label="",legend=true)
  
    display(Obs_trajectory)
    gui()
    savefig(Obs_trajectory,"/home/andreyzhitnikov/" * alg_name *string(iter) * ".pdf")
    return
end


function actionQvalues(p, t, a_id=nothing)
    mids, ∆s = [], []
    ub, lb = [], []
    for (action, ba_id) in enumerate(p.tree.nodes[1].children)
        push!(mids, (p.tree.nodes[ba_id].q + p.tree.nodes[ba_id].q_UB)/2)
        push!(∆s, (p.tree.nodes[ba_id].q - p.tree.nodes[ba_id].q_UB)/2)
        push!(ub, p.tree.nodes[ba_id].q_UB)
        push!(lb, p.tree.nodes[ba_id].q)
    end
    qbars = scatter!(mids,grid=false,yerror=∆s, m=hline, markersize=0.5, label="")
    qbars = scatter!(ub, label="upper bound")
    qbars = scatter!(lb, label="lower bound", legend=:bottomright)
    #=
    if a_id !== nothing
        ub = p.tree.nodes[a_id].q_UB
        lb = p.tree.nodes[a_id].q
        qbars = scatter!((a_id, ub), label="optimal action", color="blue")
        qbars = scatter!((a_id, lb), label="", color="blue")
    end
    =#
    xlabel!("action")
    ylabel!("Q value")
    title!("Time step $t")
    display(qbars)
    return
end