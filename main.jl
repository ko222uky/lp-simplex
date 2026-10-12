#=
Useful references / documentation: 

* Matrix / array definitions: https://docs.julialang.org/en/v1/stdlib/SparseArrays/

* HiGHS.jl:
    * https://jump.dev/JuMP.jl/stable/tutorials/getting_started/getting_started_with_JuMP/#Getting-started-with-JuMP
    * https://jump.dev/JuMP.jl/stable/packages/HiGHS/
    * https://ergo-code.github.io/HiGHS/dev/

    * Complete options at: https://ergo-code.github.io/HiGHS/dev/options/definitions/

=#
using JuMP, HiGHS
using SparseArrays, LinearAlgebra, Random, Statistics, Printf


 ###### LP PROBLEM FUNCTIONS #####
"""
This generates a random LP with n variables and m constraints. Should be guaranteed to be
feasible and bounded:
  - b = A*x0 + slack with x0 >= 0   => x0 is primal feasible
  - c = A'*y0 - s    with y0, s >= 0 => y0 is dual feasible (so the LP is bounded)

So for a given LP, we have:

|Element|Primal (b)|Dual (c)|
|---|---|---|
|Constraint| A x <= b | A'y >= c |
|Point| x0 >= 0 | y0 >= 0 |
|Slack| add s1: b = A x0 + s1 | subtract s2: c = A'y0 − s2|
|Result| x0 is primal-feasible |y0 is dual-feasible|

Worked this out from: https://en.wikipedia.org/wiki/Dual_linear_program
Where, if you have an LP with 

    * Maximize cTx subject to Ax ≤ b, x ≥ 0.

And the corresponding dual, which (in the case of primal maximization)
gives an upper bound, we have:

    * Minimize bTy subject to ATy ≥ c, y ≥ 0

Of course, if the primal is to minimize, then the dual is maximization,
and we get the lowerbound.

The actual problem lecture will take the primal as a minimization problem.
"""
function random_lp(n::Int, m::Int; density::Float64 = 0.1, rng = Random.default_rng())


    # m rows := constraints, n columns := decision variables.
    # this is the primal.
    A  = sprandn(rng, m, n, density) # we can specify the matrix density, to have "holes" in the constraints 
    
    
    x0 = rand(rng, n) # note: rand() returns a vector; rand initialized x0
                      # x0 is a primal feasible point (in primal space),
                      # since it has n variables.

    y0 = rand(rng, m) # y0 is a dual feasible point (dual space) since it has m variables,
                      # i.e., same number as there are rows/constraints.

    b  = A * x0 .+ rand(rng, m) # This is Ax + slack = b
                                # so b is the RH side of our constraints.


    c  = A' * y0 .- rand(rng, n) # A' is transpose of A; note how this is defined such 
                                 # that we have a bounded problem.
    return A,b,c
end

"""
Defines the example problem from the lecture.
Lecture 7, slide 81, on the Simplex: Tableau example.

Starting problem as n =4, m = 2 for vars, constraints, respectively.
"""
function example_lp(
    # positional args with defaults
    additional_n::Int = 0,
    additional_m::Int = 0
    ; 
    # kwargs
    density::Float64 = 0.1, 
    rng = Random.default_rng(),
    affect_orig_constraints::Bool = true)
    # we define A (mxn) with m constraints, n variables
    A = [
        1.0 2.0 -1.0 -1.0;
        -1.0 -5.0 2.0 3;
    ]

    # m-dimensional vector
    b = [1.0, 1.0]

    # n-dimensional vector, each var has cost.
    c = [7.0, 4.0, 6.0, 1.0]

    # similar to the random LP function above,
    # but interprets LP as a minimization function, so we flip the operators to their inverse, i.e.,
    # .- for .+ and vice versa.
    
    # for this, we use integers in range -20 to 20, skipping 0
    A2 = sprand(rng, additional_m, additional_n, density, (r, k) -> Float64.(rand(r, [-20:-1; 1:20], k)))
    x0 = rand(rng, 1:20, additional_n)
    y0 = rand(rng, 1:20, additional_m)
    b2 = A2 * x0 .- rand(rng, 1:20, additional_m)  # built to handle just the A2 part
    c2 = A2' * y0 .+ rand(rng, 1:20, additional_n)


    # this is like adding an independent, non-interacting addition to the original lecture problem.
    # if we want new constraints
    A_full = blockdiag(sparse(A), A2)   # easiest way to extend this.
                                        # results in a (2 + additional_m) x (4 + additional_n)
                                        # so, this is basically like:
                                        # A_full = [ A   0  ]     rows:    2 lecture constraints, then additional_m random ones
                                        #          [ 0   A2 ]     columns: 4 lecture variables,   then additional_n random ones

    # we can fill in the zeros with the following:
    # in this case, we have A_full = [A C;
    #                                B A2] # so B has additional_m rows, and b2 is length additional_m
    if affect_orig_constraints
        # I think this will not guarantee feasibility by doing this.
        # getting INFEASIBLE status...
        m1, n1 = size(A)
        C  =  sprand(rng, m1, additional_n, density, (r, k) -> Float64.(rand(r, 1:20, k)))
        B  = -sprand(rng, additional_m, n1, density, (r, k) -> Float64.(rand(r, 1:20, k))) # B is all negative

        # B  = abs.(sprandn(rng, additional_m, n1, density))
        A_full = [
            A C;
            B A2
        ]


        # # I noticed that making B positive results in DUAL_INFEASIBLE. 
        # # so, let's just fix B with the following by adjusting b2.
        # # b2 above only guarantees the new rows hold at [0; x0], which assumes the lecture vars are zero
        # # BUT the lecture variables are nonzero. So we need to think about a hypothetical point like [x_lec; x0] that satisfies constraints,
        # # or else we'll be infeasible.
        x_lec = [2.0, 0.0, 0.0, 1.0] # feasible for orig lecture vars
        b2 = b2 .+ B * x_lec         # since b2 was for A2 only and not A, this will result in infeasibility,
                                     # if we just fill in the gaps
                                     # so we adjust b2 (right side of constraint) to basically lower the target by similar amount
                                     
                                     # The top row:
                                     # A*x_lec + C*x0  >=  b

                                     # the bottom row:
                                     
                                     #      B*x_lec + A2*x0  >=  b2 new

                                     # before, we had B*x_lec + A2*x0 >= b2 old,
                                     # where B is negative. So the left side would be pulled down by this neg term.
                                     # which means in some cases  B*x_lec + A2*x0 would NOT be >= b2 old, and we'd get primal infeasibility.
                                     # but the change guarantees the bottom row satisfies A2*x0  >=  b2 old
              



    end
    b_full = [b; b2] # extends b
    c_full = [c; c2] # extends c

    return A_full, b_full, c_full
end




"""
For HiGHS see: https://ergo-code.github.io/HiGHS/stable/interfaces/julia/
For sequential, non-concurrent strats, we have:

`strategy`: 1 = dual simplex, 4 = primal simplex 

Strategies 2 and 3 should be parallelized versions.
"""
function solve_simplex(A, b, c; strategy::Int = 1, presolve::String = "off",
                       time_limit::Float64 = 600.0, threads::Int = 1, maximize::Bool = false)
    m, n = size(A)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_attribute(model, "solver", "simplex")
    set_attribute(model, "simplex_strategy", strategy)
    set_attribute(model, "presolve", presolve)
    set_attribute(model, "threads", threads) # I'll set default 1 threads. I think strats 1 and 4 can only use 1 thread.
    set_attribute(model, "time_limit", time_limit)

    if maximize
        # for maximization LPs (primal), so it
        # solves the LP from the random_lp function, which is a maximization problem.
        @variable(model, x[1:n] >= 0)
        @constraint(model, A * x .<= b)
        @objective(model, Max, dot(c, x))
    else
        # for minimization LPs (primal), so it
        # solves the LP from the lecture, which is a minimization problem.
        @variable(model, x[1:n] >= 0)
        @constraint(model, A * x .>= b)
        @objective(model, Min, dot(c, x))        
    end

    optimize!(model)

    # a vertex in n variables needs exactly n tight inequalities (x_j = 0 or row i at b_i);
    # more than n tight means the solution is (primal) degenerate.
    degenerate = false
    if has_values(model)
        xv = value.(x)


                # count(xv .< 1e-9) is num of variables that are 0 in the solution, i.e., number of tight bounds
                # so for x = [2 0 0 1], this would be 2.
                # and count(abs.(A * xv .- b) .< 1e-9) is num of constraint rows that are tight at the solution.
                # together, these are the total num of tight inequalities out of n + m.
                # if the num of tight ineq > n, we have degenerate.
        tight = count(xv .< 1e-9) + count(abs.(A * xv .- b) .< 1e-9)
        degenerate = tight > n # vertex is where n linearly independent constraints meet.
                               # but if more constraints are there than is needed to define it, we have     
                               # primal degeneracy.

                               # just note that this just means we have the same solution,
                               # but it can have multiple bases since it has extra 0s.

    end

    # the return is a named tuple.
    return (
        status = termination_status(model),    # did it complete? did it not???
        time   = solve_time(model),            # solver time only (seconds)
        iters  = simplex_iterations(model),    # equiv to the number of pivots required.
        obj    = has_values(model) ? objective_value(model) : NaN,
        x      = has_values(model) ? value.(x) : Float64[], # this is the primal solution found
        degenerate,                            # more than n tight constraints at x
    )
end

"""
Run `reps` random instances for every (n, m) pair in `sizes`.
Returns a vector of named tuples, one per solve, which we can later
use to get statistical summaries of out experiments..

Note that with `sizes`, we can define all (n,m) sizes that we want to test,
and for each of these sizes, we get our `reps` runs.

By default every rep re-solves the same instance, so the spread is runtime noise only.
Set `vary_instances = true` to draw a new random instance per rep instead.

This can be used with `random_lp`, or one that basically extends an initial LP,
such as that found in the lecture notes.
"""
function run_experiments(sizes; reps::Int = 5, density::Float64 = 0.1,
                         strategy::Int = 1, presolve::String = "on", seed::Int = 1234, 
                         solve_lecture::Bool = false, write_rep_csv::Bool = true,
                         vary_instances::Bool = false,
                         affect_orig_constraints::Bool = false, threads::Int = 1)
    # julia uses JIT compilation, so the first can be very long. We allow the
    # precompilation to happen with a warm-up.
    solve_simplex(random_lp(10, 10; density = 0.5, rng = MersenneTwister(0))...;
                strategy, presolve, maximize=true, threads=threads)

    if !solve_lecture
        # vector of named tuples; it's like an immutable dict, basically...
        results = NamedTuple[]
        @printf("%8s %8s %5s %7s %12s %12s %14s %6s  %s  %s\n", "n", "m", "rep", "density", "time (s)", "iterations", "obj", "degen", "status", "x")

        # runs our replicates for a given (n,m) problem.
        for (n, m) in sizes, rep in 1:reps
            rng = MersenneTwister(hash((seed, n, m, vary_instances ? rep : 0)))   # same instance every rep unless vary_instances
            A, b, c = random_lp(n, m; density, rng)
            r = solve_simplex(A, b, c; strategy, presolve, maximize=true)
            push!(results, (; n, m, rep, density, r...)) # the ';' is a way of forcing named fields, akin to python's **
            @printf("%8d %8d %5d %7.2f %12.4f %12d %14.6g %6s  %s  %s\n", n, m, rep, density, r.time, r.iters, r.obj, r.degenerate, r.status, round.(r.x; digits = 4))
        end
    else
        # n_init = 4 and n_init=2 is predefined.
        n_init = 4
        m_init = 2
        # vector of named tuples; it's like an immutable dict, basically...
        results = NamedTuple[]
        @printf("%8s %8s %5s %7s %12s %12s %14s %6s  %s  %s\n", "n", "m", "rep", "density", "time (s)", "iterations", "obj", "degen", "status", "x")

        # runs our replicates for a given (n,m) problem.
      
        for (additional_n, additional_m) in sizes, rep in 1:reps
            rng = MersenneTwister(hash((seed, additional_n + n_init, additional_m + m_init, vary_instances ? rep : 0)))   # same instance every rep unless vary_instances
            A, b, c = example_lp(additional_n, additional_m; density, rng, affect_orig_constraints)
           
            printstyled("\nMatrix A :\n\n", A, color=:green)
            println("\n\nVector b :\n", b)
            println("\n\nVector c :\n", c, "\n\n")

    
            r = solve_simplex(A, b, c; strategy, presolve, maximize=false) # so we minimize
            push!(results, (; n = additional_n + n_init, m = additional_m + m_init, rep, density, r...)) # the ';' is a way of forcing named fields, akin to python's **
            @printf("%8d %8d %5d %7.2f %12.4f %12d %14.6g %6s  %s  %s\n", additional_n + n_init, additional_m + m_init, rep, density, r.time, r.iters, r.obj, r.degenerate, r.status, round.(r.x; digits = 4))
        end
    end

    # return our results (expierment replicates for a given experiment)
    return results
end

"""
Prints out statistics of experimental runs.
If `path` is given, the same summary table is also written to that CSV file.
"""
function print_summary(results; path = nothing)
    printstyled("\nSummary (mean over repetitions)\n"; color=:green)
    @printf("%8s %8s %7s %12s %12s %14s\n", "n", "m", "density", "mean time", "std time", "mean iters")
    
    # need notes here else I'll forget the syntax
    # opens csv if and only if we have a path
    io = path === nothing ? nothing : open(path, "w")
    
    # this is like javascript's '==='. if io is nothing, then evaluates to true, and we don't run right side.
    # Otherwise run the right side, which writes to CSV
    io === nothing || println(io, "n,m,density,mean_time,std_time,mean_iters")

    for key in unique((r.n, r.m, r.density) for r in results) # for each unique (n, m, density) to avoid duplicate prints
        rs = filter(r -> (r.n, r.m, r.density) == key, results)
        t  = [r.time for r in rs]
        it = [r.iters for r in rs]
        sd = length(t) > 1 ? std(t) : 0.0
        @printf("%8d %8d %7.2f %12.6f %12.6f %14.1f\n", key[1], key[2], key[3], mean(t), sd, mean(it))

        # again, if we have valid io, then write the results to the csv, too.
        io === nothing || println(io, join((key[1], key[2], key[3], mean(t), sd, mean(it)), ","))
    end

    # close if io is valid...
    io === nothing || close(io)
end


"""
Writes the full output of every replicate (one row per solve), including
the degeneracy flag and the solution vector x.
"""
function save_reps_csv(results, path)
    open(path, "w") do io
        println(io, "n,m,rep,density,status,time,iters,obj,degenerate,x")
        for r in results
            x = "\"" * join(r.x, " ") * "\""
            println(io, join((r.n, r.m, r.rep, r.density, r.status, r.time, r.iters, r.obj, r.degenerate, x), ","))
        end
    end
end

# ---------------------------------------------------------------------------
# Running the actual experiments
# ---------------------------------------------------------------------------

######################################################
# Dummy Experiment - for testing
######################################################
# dummy example with m = n / 2, just to test the behavior of the functions
# and see that they run.
# sizes = [(n, n ÷ 2) for n in (100, 200, 400, 800)]
# println("Testing sizes (n vars, m constraints): ", sizes)

# printstyled("Running dummy experiments...\n"; color=:yellow)
# results = run_experiments(sizes; reps = 5, density = 0.5, strategy = 1, presolve = "on", vary_instances=false)
# printstyled("Done!"; color=:green)

# print_summary(results; path = "dummy_summary.csv")
# save_reps_csv(results, "dummy_reps.csv")

######################################################
# Lecture LP Experiment 
######################################################
additional_ns = [0, 6, 16, 26, 36, 46] # equivalent to n = 4, 10, 20, 30, 40, 50
additional_ms = [0, 4, 8, 12] # equivalent to m = 2, 6, 10, 14
additional_sizes = vec(collect(Iterators.product(additional_ns, additional_ms)))   # 24 tuples
println("Testing extensions of the lecture example LP: ", additional_sizes)

printstyled("Running lecture experiments...\n"; color=:yellow)
results = run_experiments(
    additional_sizes; reps = 5, 
    density = 0.5 # 0.2, # the density of the additional matrix A2 used to extend A...
    ,strategy = 1  # should be 1 for the dual method.
    ,threads=1     # only applicable for strategy 2 or 3, but I failed to get those to work.
    ,presolve = "off",
    solve_lecture=true # ADD THIS TO SOLVE LECTURE LP
    ,vary_instances=false # the replicate
    ,affect_orig_constraints=true
)
printstyled("Done!"; color=:green)
print_summary(results; path = "summary.csv")
save_reps_csv(results, "reps.csv")



