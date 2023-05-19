import gymnasium as gym

class Env():
    def __init__(self, env_name: str, render_mode: str = None):
        self.env_name = env_name
        self.render_mode = render_mode
    
    def create_env(self):
        try:
            env = gym.make(
                id=self.env_name,
                render_mode=self.render_mode
            )
            return env

        except gym.error.UnregisteredEnv as e:
            print(f"Error: The environment '{self.env_name}' is not registered in OpenAI Gym.")
            print("Check the spelling or try another environment name.")
            print(f"Original error message: {str(e)}")
            return None

        except gym.error.Error as e:
            print(f"Error: An error occurred while creating the environment '{self.env_name}'.")
            print(f"Original error message: {str(e)}")
            return None
