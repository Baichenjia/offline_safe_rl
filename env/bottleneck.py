try:
    from flow.core.experiment import Experiment
    from flow.utils.registry import make_create_env

    def create_bottleneck_env():
        exp_config = "singleagent_bottleneck"
        # Get the flow_params object.
        module = __import__("examples.exp_configs.rl.singleagent", fromlist=[exp_config])
        flow_params = getattr(module, exp_config).flow_params

        flow_params['sim'].render = False
        flow_params['simulator'] = 'traci'

        create_env, _ = make_create_env(flow_params)
        return create_env()

except ModuleNotFoundError:
    import warnings
    warnings.warn("FLOW not found during import, bottleneck env is not loaded")

    def create_bottleneck_env():
        return None