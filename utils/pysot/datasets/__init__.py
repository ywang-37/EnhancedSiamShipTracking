from .vot import VOTDataset


class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name 'VOT2018', 'VOT2016'
            dataset_root: dataset root
        Return:
            dataset
        """
        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        if 'VOT2018' == name or 'VOT2016' == name:
            dataset = VOTDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

