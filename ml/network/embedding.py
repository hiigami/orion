import numpy as np


class Embedding(object):

    def __init__(self, partition_strategy='mod'):
        self._partition_strategy = partition_strategy

    def _ps_mod(self, ids, num_partitions):
        return ids % num_partitions, ids // num_partitions

    @staticmethod
    def scalar_partition():
        # TODO (hiigami) complete method
        pass

    @staticmethod
    def vector_partition(data, partition, ids_shape):
        shape = (partition.shape[0], data.shape[1])
        output = np.empty(shape, dtype=data.dtype)
        el = 0
        for index in partition:
            output[el] = data[index]
            el += 1
        return output.reshape(ids_shape)

    def look_up(self, lookup_table, ids):
        num_partitions = lookup_table.size
        _ids = ids.flatten()
        partitions, new_ids = None, None
        if self._partition_strategy == 'mod':
            partitions, new_ids = self._ps_mod(_ids, num_partitions)
        else:
            # TODO (hiigami) add more partition options
            raise NotImplementedError("{0}._ps_{1}".format(self.__class__.__name__,
                                                           self._partition_strategy))
        partitions = partitions.astype(np.int32)
        _shape = ids.shape + lookup_table.shape[1:]
        partitions = self.vector_partition(lookup_table, partitions, _shape)
        return partitions
