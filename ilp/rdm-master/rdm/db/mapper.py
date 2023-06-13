# Mapping unseen relational examples to an existing propositionalized domain
import tempfile
import subprocess
import os
import re
import arff

from .converters import RSDConverter, TreeLikerConverter


def _feature_numbers(features):
    n = len(features.splitlines())
    featureIDs = [str(id) for id in range(1, n+1)]
    return 'featureIDs([%s]).' % (','.join(featureIDs))


example_id_pattern = re.compile(r', (?P<id>.+)\)\.')
def _example_ids(pred, examples):
    exampleIDs = example_id_pattern.findall(examples, re.M)
    return '%s([%s]).' % (pred, ','.join(exampleIDs))


def domain_map(features, feature_format, train_context, test_context,
               intervals={},
               format='arff',
               positive_class=None):
    '''
    Use the features returned by a propositionalization method to map
    unseen test examples into the new feature space.

      :param features: string of features as returned by rsd, aleph or treeliker
      :param feature_format: 'rsd', 'aleph', 'treeliker'
      :param train_context: DBContext with training examples
      :param test_context: DBContext with test examples
      :param intervals: discretization intervals (optional)
      :param format: output format (only arff is used atm)
      :param positive_class: required for aleph

      :return: returns the test examples in propositional form
      :rtype: str

      :Example:

      >>> test_arff = mapper.domain_map(features, 'rsd', train_context, test_context)
    '''
    if feature_format == 'aleph' and positive_class is None:
        raise ValueError("positive_class parameter is required for 'aleph' feature format")
    
    dataset = None
    if feature_format in ['rsd', 'aleph']:
        train_rsd = RSDConverter(train_context)
        test_rsd = RSDConverter(test_context, discr_intervals=intervals)
        mapper_target_name = train_context.target_table + '_mapper'
        train_examples = train_rsd.all_examples(pred_name=mapper_target_name)
        test_examples = test_rsd.all_examples(pred_name=mapper_target_name)

        if feature_format == 'aleph':
            features = aleph_to_rsd_features(features)

        prolog_bk = '\n'.join([
            _example_ids('testExampleIDs', test_examples),
            '%% test examples',
            test_examples,
            '%% train examples',
            train_examples,
            '%% train background knowledge',
            train_rsd.background_knowledge(),
            '%% test background knowledge',
            test_rsd.background_knowledge(),
            _feature_numbers(features),
            '%% features',
            features,
        ])
        THIS_DIR = os.path.dirname(__file__) if os.path.dirname(__file__) else '.'
        f = tempfile.NamedTemporaryFile(delete=False, mode='w')
        f.write(prolog_bk)
        f.close()
        cmd_args = ['yap', '-L', '--', '%s/mapper.pl' % THIS_DIR, f.name, mapper_target_name]
        evaluations = subprocess.check_output(cmd_args).decode()
        dataset = dump_dataset(features, feature_format, evaluations,
                               train_context,
                               format=format,
                               positive_class=positive_class)

        # Cleanup
        os.remove(f.name)

    elif feature_format == 'treeliker':
        # We provide treeliker with the test dataset
        # since it has a built-in ability to evaluate features
        treeliker_test = TreeLikerConverter(test_context,
                                            discr_intervals=intervals)
        treeliker = features
        treeliker.test_dataset = treeliker_test.dataset()
        _, test_dataset = treeliker.run()

        if format == 'arff':
            dataset = test_dataset
        else:
            return 'unsupported format'

    return dataset


def dump_dataset(features, feature_format, evaluations, train_context,
                 format='arff',
                 positive_class=None):
    if format == 'arff':
        data = {
            'attributes': [],
            'data': [],
            'description': '',
            'relation': 'default'
        }
        n_features = len(features.splitlines())
        for i in range(1, n_features + 1):
            feature = ('f%d' % i, ['+', '-'])
            data['attributes'].append(feature)

        target = train_context.target_table
        if not target in train_context.orng_tables:
            raise Exception('Target table is not preloaded in memory! Please select the `dump data` parameter in the converter widget.')
        if feature_format == 'aleph':
            target_vals = ('negative', 'positive')
        else:
            orng_target = train_context.orng_tables[target]
            target_vals = tuple(sorted(orng_target.domain.class_var.values))
        class_attr = ('class', target_vals)
        data['attributes'].append(class_attr)
        for line in evaluations.splitlines():
            values = line.strip().split()
            if feature_format == 'aleph':
                class_val = values[-1]
                if class_val == positive_class:
                    values[-1] = 'positive'
                else:
                    values[-1] = 'negative'
            data['data'].append(values)
        return arff.dumps(data)

    elif format == 'csv':
        data = ''
        for line in evaluations.splitlines():
            values = line.strip().split()
            data = data + ','.join(values) + '\n'
        return data

    return 'unsupported format'


def aleph_to_rsd_features(features):
    converted_features = []
    for line in features.splitlines():
        if not line.startswith('feature'):
            continue

        feature_id = len(converted_features) + 1
        feature_body = line[line.find(':-'):line.find(')).')] + '.'
        new_feature = 'f(%d, A)%s' % (feature_id, feature_body)
        converted_features.append(new_feature)

    return '\n'.join(converted_features)
