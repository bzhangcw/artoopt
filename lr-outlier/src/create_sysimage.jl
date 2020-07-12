# ....................
# @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
# @project: src
# @file: /create_sysimage.jl
# @created: Wednesday, 8th April 2020
# @author: brentian (chuwzhang@126.com)
# @modified: brentian (chuwzhang@126.com>)
#    Wednesday, 8th April 2020 7:15:24 pm
# ....................
# @description: 
Base.init_depot_path()
Base.init_load_path()

@eval Module() begin
    Base.include(@__MODULE__, "main.jl")
    for (pkgid, mod) in Base.loaded_modules
        if !(pkgid.name in ("Main", "Core", "Base"))
            eval(@__MODULE__, :(const $(Symbol(mod)) = $mod))
        end
    end
    for statement in readlines("app_precompile.jl")
        try
            Base.include_string(@__MODULE__, statement)
        catch
            # See julia issue #28808
            Core.println("failed to compile statement: ", statement)
        end
    end
end # module

empty!(LOAD_PATH)
empty!(DEPOT_PATH)
