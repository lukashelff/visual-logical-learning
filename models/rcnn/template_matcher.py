from michalski_trains.dataset import blender_categories, rcnn_blender_categories
from util import *

def rcnn_to_car_number(label_val):
    return label_val - len(blender_categories()) + 1
def prediction_to_symbolic(prediction, threshold=.8):
    '''
    Convert prediction to symbolic representation
    :param prediction: prediction from model
    :param threshold: threshold for similarity between two segments
    :return: symbolic representation of the scene
    '''
    labels = prediction["labels"]
    boxes = prediction["boxes"]
    masks = prediction["masks"]
    scores = prediction["scores"].to('cpu').tolist()
    cars = sorted(labels[labels >= len(blender_categories())].unique())
    # labels = labels.tolist()
    label_names = []
    for l in labels:
        cats = rcnn_blender_categories()
        if l < len(cats):
            label_names.append(cats[l])
        else:
            label_names.append(f'car_{rcnn_to_car_number(l)}')

    # get indices of all cars
    all_car_indices = []
    # select valid cars
    selected_car_indices = []
    debug_info = ""
    # get all cars, select car with highest score if there are multiple cars with same car number, others are discarded
    for car in cars:
        indices = ((labels == car).nonzero(as_tuple=True)[0])
        indices = indices.to('cpu').tolist()
        all_car_indices += indices
        if len(indices) > 1:
            # select car with the highest score if there are multiple cars with same car number
            debug_info += f"Multiple cars with same number: {len(indices)} cars with car number {rcnn_to_car_number(car)}." \
                          f" Selecting car with highest prediction score."
            idx = indices[0]
            # issues = True
            for i in indices[1:]:
                if scores[i] > scores[idx]:
                    idx = i
        else:
            idx = indices[0]
        selected_car_indices.append(idx)
    # get indices of all attributes
    attribute_indices = [i for i in range(len(labels)) if i not in all_car_indices]

    shape = ['rectangle', 'bucket', 'ellipse', 'hexagon', 'u_shaped']
    length = ['short', 'long']
    walls = ["double", 'not_double']
    roofs = ['arc', 'flat', 'jagged', 'peaked']
    wheel_count = ['2', '3']
    load_obj = ["rectangle", "triangle", 'circle', 'diamond', 'hexagon', 'utriangle']
    original_categories = ['none'] + shape + length + walls + roofs + wheel_count + load_obj
    # initialize symbolic representation
    car_init = [1, 6, 8, 0, 14, 0, 0, 0]
    train = torch.tensor(car_init * len(cars), dtype=torch.uint8)
    # train = torch.zeros(len(cars) * 8, dtype=torch.uint8)
    train_scores = [0] * len(cars) * 8
    skipped_indicies = []
    # iterate over all predicted attributes
    for attribute_index in attribute_indices:
        allocated = False
        label = labels[attribute_index]
        label_name = label_names[attribute_index]
        car_similarities = get_all_similarities_score(masks, car_indices=selected_car_indices,
                                                      attribute_index=attribute_index)
        if len(car_similarities) == 0:
            debug_info += f"Attribute {label_name} not allocated to any car. "
            continue
        car_number = np.argmax(car_similarities) + 1
        similarity = car_similarities[car_number - 1]
        # if similarity higher than threshold allocate attribute to car
        if similarity > threshold:
            # class_int = blender_categories().index(label_name)
            # binary_class = np.zeros(22)
            # binary_class[label] = 1
            label_category = class_to_label(label)
            idx = (car_number - 1) * 8 + label_category
            # if attribute is a payload, check if there is already a payload allocated to the car
            if label_category == 5:
                # todo: sort payload by score to replace payload with lower score if there are to many payloads
                while train[idx] != 0 and (idx % 8) < 7:
                    idx += 1
            if train[idx] != 0:
                if train[idx] == label:
                    debug_info += f"Duplicate Mask: Mask for car {car_number} with label {label_name} was predicted twice."
                elif train[idx] != label:
                    if scores[attribute_index] > train_scores[idx]:
                        debug_info += f'Mask conflict: {blender_categories()[train[idx]]} with score' \
                                      f' {round(train_scores[idx], 3)} and {blender_categories()[label]} with score' \
                                      f' {round(scores[attribute_index], 3)} for car {car_number}. Selecting label with higher score.'
                        train[idx] = label
                        allocated = True
                        train_scores[idx] = scores[attribute_index]
                    else:
                        debug_info += f'Mask conflict: {blender_categories()[train[idx]]} with score ' \
                                      f'{round(train_scores[idx], 3)} and {blender_categories()[label]} with score' \
                                      f' {round(scores[attribute_index], 3)} for car {car_number}. Selecting label with higher score.'
            else:
                train[idx] = label
                allocated = True
                train_scores[idx] = scores[attribute_index]
            # break
        if not allocated:
            skipped_indicies.append(attribute_index)
    nr_attr_processed = len(train[(train > 0) & (train < 22)])
    model_label_prdictions = prediction['labels'].to('cpu').numpy()
    nr_output_attr = len(model_label_prdictions[(model_label_prdictions > 0) & (model_label_prdictions < 22)])
    debug_info = f'Number of predictions: {nr_output_attr}, number of found allocations: {nr_attr_processed}, ' \
                 f'not allocated label number: {skipped_indicies}. ' \
                 + debug_info
    return train, debug_info


def get_all_similarities_score(masks, attribute_index, car_indices):
    car_similarities = []
    for car_n, car_index in enumerate(car_indices):
        whole_car_mask = masks[car_index]
        mask = masks[attribute_index]
        similarity = get_similarity_score(mask, whole_car_mask)
        car_similarities += [similarity.item()]
    return car_similarities


def get_similarity_score(mask, whole_car_mask):
    # determine to which degree mask is included in whole car mask
    # calculate similarity value by summing up all values in mask where mask is smaller than whole car mask
    # and summing up all values of whole car mask where mask is higher than whole car mask
    similarity = mask[mask <= whole_car_mask].sum() + whole_car_mask[mask > whole_car_mask].sum()
    similarity = similarity / mask.sum()

    # calculate difference between mask and whole car mask for values where mask is higher than whole car mask
    # asimilarity = mask[mask > whole_car_mask].sum() - whole_car_mask[mask > whole_car_mask].sum()
    # similarity = 1 - asimilarity / mask.sum()

    # calculate similarity by multiplication of mask and whole car mask, problem because we hve float values
    # when mask = whole car mask = 0.3 => similarity = 0.3 * 0.3 = 0.09 => similarity is too low
    # simi = mask * whole_car_mask
    # similarity = simi.sum() / mask.sum()
    return similarity