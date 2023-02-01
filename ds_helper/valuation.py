import torch
import torch.nn as nn
import torch.nn.functional as F


class MichalskiValuationModule(nn.Module):
    """A module to call valuation functions.
        Attrs:
            lang (language): The language.
            device (device): The device.
            layers (list(nn.Module)): The list of valuation functions.
            vfs (dic(str->nn.Module)): The dictionary that maps a predicate name to the corresponding valuation function.
            attrs (dic(term->tensor)): The dictionary that maps an attribute term to the corresponding one-hot encoding.
            dataset (str): The dataset.
    """

    def __init__(self, lang, device, pretrained=True):
        super().__init__()
        self.lang = lang
        self.device = device
        self.car_nums = ['1', '2', '3', '4']
        self.colors = ["yellow", "green", "grey", "red", "blue"]
        self.lengths = ["short", "long"]
        self.walls = ["full", "braced"]
        self.roofs = ["none", "foundation", "solid_roof", "braced_roof", "peaked_roof"]
        self.wheels = ['2', '3']
        self.loads = ["blue_box", "golden_vase", "barrel", "diamond", "metal_box"]
        self.load_nums = ['0', '1', '2', '3']
        self.obj_desc = {
            'car_num': self.car_nums,
            'color': self.colors,
            'length': self.lengths,
            'wall': self.walls,
            'roof': self.roofs,
            'wheel': self.wheels,
            'load': self.loads,
            'load_num': self.load_nums,
        }

        self.layers, self.vfs = self.init_valuation_functions(
            device, pretrained)

    def init_valuation_functions(self, device, pretrained):
        """
            Args:
                device (device): The device.
                pretrained (bool): The flag if the neural predicates are pretrained or not.

            Retunrs:
                layers (list(nn.Module)): The list of valuation functions.
                vfs (dic(str->nn.Module)): The dictionaty that maps a predicate name to the corresponding valuation function.
        """
        vfs = {}  # pred name -> valuation function
        v_in = MichalskiInValuationFunction(device)
        v_car_num = MichalskiCarNumValuationFunction(device)
        v_color = MichalskiColorValuationFunction(device)
        v_length = MichalskiLengthValuationFunction(device)
        v_wall = MichalskiWallValuationFunction(device)
        v_roof = MichalskiRoofValuationFunction(device)
        v_wheel = MichalskiWheelValuationFunction(device)
        v_load = MichalskiLoadValuationFunction(device)
        v_load_num = MichalskiLoadNumValuationFunction(device)
        vfs['in'] = v_in
        vfs['car_num'] = v_car_num
        vfs['color'] = v_color
        vfs['length'] = v_length
        vfs['wall'] = v_wall
        vfs['roof'] = v_roof
        vfs['wheel'] = v_wheel
        vfs['load'] = v_load
        vfs['load_num'] = v_load_num

        # if pretrained:
        #     vfs['rightside'].load_state_dict(torch.load(
        #         'src/weights/neural_predicates/rightside_pretrain.pt', map_location=device))
        #     vfs['rightside'].eval()
        #     vfs['leftside'].load_state_dict(torch.load(
        #         'src/weights/neural_predicates/leftside_pretrain.pt', map_location=device))
        #     vfs['leftside'].eval()
        #     vfs['front'].load_state_dict(torch.load(
        #         'src/weights/neural_predicates/front_pretrain.pt', map_location=device))
        #     vfs['front'].eval()
        #     print('Pretrained  neural predicates have been loaded!')
        return nn.ModuleList([v_in, v_car_num, v_color, v_length, v_wall, v_roof, v_load, v_load_num]), vfs

    def forward(self, zs, atom):
        """Convert the object-centric representation to a valuation tensor.

            Args:
                zs (tensor): The object-centric representaion (the output of the YOLO model).
                atom (atom): The target atom to compute its proability.

            Returns:
                A batch of the probabilities of the target atom.
        """
        # term: logical term
        # arg: vector representation of the term
        # zs = self.preprocess(zs)
        args = [self.ground_to_tensor(term, zs) for term in atom.terms]
        # call valuation function
        return self.vfs[atom.pred.name](*args)

    def ground_to_tensor(self, term, zs):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                zs (tensor): The object-centric representation.
        """
        term_index = self.lang.term_index(term)
        if term.dtype.name == 'car':
            return zs[:, term_index]
        elif term.dtype.name == 'image':
            return None
        else:
            # other attributes
            return self.term_to_onehot(term, batch_size=zs.size(0))

    def term_to_onehot(self, term, batch_size):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                zs (tensor): The object-centric representation.

            Return:
                The tensor representation of the input term.
        """
        within = term.dtype.name in self.obj_desc
        if term.dtype.name in self.obj_desc:
            values = self.obj_desc[term.dtype.name]
            val_name = term.name
            try:
                val_idx = values.index(val_name)
            except:
                raise AttributeError(f'value {val_name} is not an attribute of {term.dtype.name}: {values}')
            return self.to_onehot_batch(val_idx, len(values), batch_size)
        else:
            assert True, 'Invalid term: ' + str(term)

    def to_onehot_batch(self, i, length, batch_size):
        """Compute the one-hot encoding that is expanded to the batch size.
        """
        onehot = torch.zeros(batch_size, length, ).to(self.device)
        onehot[:, i] = 1.0
        return onehot


##########################################
# Michalski valuation functions for slot attention #
##########################################


class MichalskiInValuationFunction(nn.Module):
    """The function v_in.
    """

    def __init__(self, device):
        super(MichalskiInValuationFunction, self).__init__()

    def forward(self, z, x):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + car_number + color + length + wall + roof + load + load_number
                [objectness, 1, 2, 3, 4, yellow, green, grey, red, blue, short, long, full, braced,
                 none, foundation, solid_roof, braced_roof, peaked roof, 2,3,
                 blue_box, golden_vase, barrel, diamond, metal_box, 0,1,2,3]
            x (none): A dummy argument to represent the input constant.

        Returns:
            A batch of probabilities.
        """
        # return the objectness
        return z[:, 0]


class MichalskiCarNumValuationFunction(nn.Module):
    """The function v__car_num.
    """

    def __init__(self, device):
        super(MichalskiCarNumValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + car_number + color + length + wall + roof + load + load_number
                [objectness, 1, 2, 3, 4, yellow, green, grey, red, blue, short, long, full, braced,
                 none, foundation, solid_roof, braced_roof, peaked roof, 2,3,
                 blue_box, golden_vase, barrel, diamond, metal_box, 0,1,2,3]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_shape = z[:, 1:5]
        return (a * z_shape).sum(dim=1)


class MichalskiColorValuationFunction(nn.Module):
    """The function v_size.
    """

    def __init__(self, device):
        super(MichalskiColorValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + car_number + color + length + wall + roof + load + load_number
                [objectness, 1, 2, 3, 4, yellow, green, grey, red, blue, short, long, full, braced,
                 none, foundation, solid_roof, braced_roof, peaked roof, 2,3,
                 blue_box, golden_vase, barrel, diamond, metal_box, 0,1,2,3]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_size = z[:, 5:10]
        return (a * z_size).sum(dim=1)


class MichalskiLengthValuationFunction(nn.Module):
    """The function v_material.
    """

    def __init__(self, device):
        super(MichalskiLengthValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + car_number + color + length + wall + roof + load + load_number
                [objectness, 1, 2, 3, 4, yellow, green, grey, red, blue, short, long, full, braced,
                 none, foundation, solid_roof, braced_roof, peaked roof, 2,3,
                 blue_box, golden_vase, barrel, diamond, metal_box, 0,1,2,3]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_material = z[:, 10:12]
        return (a * z_material).sum(dim=1)


class MichalskiWallValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self, device):
        super(MichalskiWallValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + car_number + color + length + wall + roof + load + load_number
                [objectness, 1, 2, 3, 4, yellow, green, grey, red, blue, short, long, full, braced,
                 none, foundation, solid_roof, braced_roof, peaked roof, 2,3,
                 blue_box, golden_vase, barrel, diamond, metal_box, 0,1,2,3]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_color = z[:, 12:14]
        return (a * z_color).sum(dim=1)


class MichalskiRoofValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self, device):
        super(MichalskiRoofValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + car_number + color + length + wall + roof + load + load_number
                [objectness, 1, 2, 3, 4, yellow, green, grey, red, blue, short, long, full, braced,
                 none, foundation, solid_roof, braced_roof, peaked roof, 2,3,
                 blue_box, golden_vase, barrel, diamond, metal_box, 0,1,2,3]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_color = z[:, 14:19]
        return (a * z_color).sum(dim=1)


class MichalskiWheelValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self, device):
        super(MichalskiWheelValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + car_number + color + length + wall + roof + load + load_number
                [objectness, 1, 2, 3, 4, yellow, green, grey, red, blue, short, long, full, braced,
                 none, foundation, solid_roof, braced_roof, peaked roof, 2,3,
                 blue_box, golden_vase, barrel, diamond, metal_box, 0,1,2,3]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_color = z[:, 19:21]
        return (a * z_color).sum(dim=1)


class MichalskiLoadValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self, device):
        super(MichalskiLoadValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + car_number + color + length + wall + roof + load + load_number
                [objectness, 1, 2, 3, 4, yellow, green, grey, red, blue, short, long, full, braced,
                 none, foundation, solid_roof, braced_roof, peaked roof, 2,3,
                 blue_box, golden_vase, barrel, diamond, metal_box, 0,1,2,3]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_color = z[:, 21:26]
        return (a * z_color).sum(dim=1)


class MichalskiLoadNumValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self, device):
        super(MichalskiLoadNumValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + car_number + color + length + wall + roof + load + load_number
                [objectness, 1, 2, 3, 4, yellow, green, grey, red, blue, short, long, full, braced,
                 none, foundation, solid_roof, braced_roof, peaked_roof, 2,3,
                 blue_box, golden_vase, barrel, diamond, metal_box, 0,1,2,3]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_color = z[:, 26:]
        return (a * z_color).sum(dim=1)
