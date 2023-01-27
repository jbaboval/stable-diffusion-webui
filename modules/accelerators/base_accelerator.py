import abc

class BaseAccelerator(abc.ABC):

    def implements(self, op):
        return hasattr(self, op) and callable(getattr(self, op))

    @abc.abstractmethod
    def get_device(self):
        raise Exception("Not Implemented!")
        return

    #@abc.abstractmethod
    def get_available_vram(self):
        raise Exception("Not Implemented!")
        return

    @abc.abstractmethod
    def memory_stats(self):
        raise Exception("Not Implemented!")
        return
    
    @abc.abstractmethod
    def memory_summary(self):
        raise Exception("Not Implemented!")
        return
    
    #@abc.abstractmethod
    def mem_get_info(self):
        raise Exception("Not Implemented!")
        return
    
    #@abc.abstractmethod
    def get_free_memory(self):
        raise Exception("Not Implemented!")
        return
    
    @abc.abstractmethod
    def get_total_memory(self):
        raise Exception("Not Implemented!")
        return
    
    @abc.abstractmethod
    def reset_peak_memory_stats(self):
        raise Exception("Not Implemented!")
        return

    @abc.abstractmethod
    def empty_cache(self):
        raise Exception("Not Implemented!")
        return

    @abc.abstractmethod
    def gc(self):
        raise Exception("Not Implemented!")
        return
    
    @abc.abstractmethod
    def enable_tf32(self):
        raise Exception("Not Implemented!")
        return

    @abc.abstractmethod
    def get_rng_state_all(self):
        raise Exception("Not Implemented!")
    
    @abc.abstractmethod
    def set_rng_state(self, state):
        raise Exception("Not Implemented!")

    @abc.abstractmethod
    def manual_seed(self, seed):
        raise Exception("Not Implemented!")

    @abc.abstractproperty
    def amp(self):
        raise Exception("Not Implemented!")
        return

