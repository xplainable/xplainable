from numpy import isnan, nan
from numpy.typing import ArrayLike


class DualDict:

    def __init__(self, dictionary: dict = None, reversed=False):
        if dictionary is not None:
            if reversed:
                self._reverse = dictionary
                self._forward = {v: i for (i, v) in dictionary.items()}
            else:
                self._forward = dictionary
                self._reverse = {v: i for (i, v) in dictionary.items()}
        else:
            self._forward = dict()
            self._reverse = dict()

    @property
    def forward(self):
        return self._forward.copy()

    @property
    def reverse(self):
        return self._reverse.copy()

    def __getitem__(self, key):
        if key in self._forward:
            return self._forward[key]
        elif key in self._reverse:
            return self._reverse[key]
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        if key in self._reverse:
            self.set_item_directional(key, value, True)
        else:
            self.set_item_directional(key, value, False)

    def get_item_directional(self, key, reverse=False):
        if reverse:
            return self._reverse[key]
        else:
            return self._forward[key]

    def set_item_directional(self, key, value, reverse=False):
        old = None
        if reverse:
            if key in self._reverse:
                old = self._reverse[key]
                self._forward.pop(old)
            self._reverse[key] = value
            self._forward[value] = key
        else:
            if key in self._forward:
                old = self._forward[key]
                self._reverse.pop(old)
            self._forward[key] = value
            self._reverse[value] = key
        return old

    def __iter__(self, reverse=False):
        for key in self.keys(reverse):
            yield key

    def items(self):
        return [(k, self._forward[k]) for k in self.keys()]

    def __repr__(self, reverse=False):
        if reverse:
            return str(self._reverse)
        else:
            return str(self._forward)

    def __contains__(self, key):
        return key in self._forward or key in self._reverse

    def keys(self, reversed=False):
        if reversed:
            return self.values()
        else:
            return self._forward.keys()

    def values(self, reversed=False):
        if reversed:
            return self.keys()
        else:
            return self._forward.values()

    def __len__(self):
        return len(self._forward)


class FeatureMap(DualDict):

    def __init__(self, dictionary: dict = None, reversed=False):
        super().__init__(dictionary, reversed)

    def __getitem__(self, key):
        if key in self._forward:
            return self._forward[key]
        elif key in self._reverse:
            return self._reverse[key]
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        if type(key) == ArrayLike:
            if isnan(key) or key == "Null": return nan
        if key in self._reverse:
            self.set_item_directional(key, value, True)
        else:
            self.set_item_directional(key, value, False)

    def get_item_directional(self, key, reverse=False):
        if reverse:
            if isnan(key):
                return "Null"
            return self._reverse[key]
        else:
            if key == "Null":
                return nan
            return self._forward[key]

    def set_item_directional(self, key, value, reverse=False):
        old = None
        if reverse:
            if isnan(key):
                return "Null"
            if key in self._reverse:
                old = self._reverse[key]
                self._forward.pop(old)
            self._reverse[key] = value
            self._forward[value] = key
        else:
            if key == "Null":
                return nan
            if key in self._forward:
                old = self._forward[key]
                self._reverse.pop(old)
            self._forward[key] = value
            self._reverse[value] = key
        return old

    def __repr__(self, reverse=False):
        if reverse:
            out_dict = {nan: "Null"}
            out_dict.update(self._reverse.copy())
        else:
            out_dict = {"Null": nan}
            out_dict.update(self._forward.copy())
        return str(out_dict)


class TargetMap(DualDict):

    def __init__(self, dictionary: dict = None, reversed=False):
        super().__init__(dictionary, reversed)

    def __getitem__(self, key):
        if key in self._forward:
            return self._forward[key]
        elif key in self._reverse:
            return self._reverse[key]
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        if isnan(key) or key == "Null": return -1
        if key in self._reverse:
            self.set_item_directional(key, value, True)
        else:
            self.set_item_directional(key, value, False)

    def get_item_directional(self, key, reverse=False):
        if reverse:
            if isnan(key):
                return "Null"
            return self._reverse[key]
        else:
            if key == "Null":
                return nan
            return self._forward[key]

    def set_item_directional(self, key, value, reverse=False):
        old = None
        if reverse:
            if isnan(key):
                return "Null"
            if key in self._reverse:
                old = self._reverse[key]
                self._forward.pop(old)
            self._reverse[key] = value
            self._forward[value] = key
        else:
            if key == "Null":
                return nan
            if key in self._forward:
                old = self._forward[key]
                self._reverse.pop(old)
            self._forward[key] = value
            self._reverse[value] = key
        return old

    def __repr__(self, reverse=False):
        if reverse:
            out_dict = {nan: "Null"}
            out_dict.update(self._reverse.copy())
        else:
            out_dict = {"Null": nan}
            out_dict.update(self._forward.copy())
        return str(out_dict)


