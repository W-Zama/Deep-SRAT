class Config:
    debug_mode = False

    @staticmethod
    def set_debug_mode(debug):
        Config.debug_mode = debug

    @staticmethod
    def is_debug_mode():
        return Config.debug_mode
