#
# Author: matic.perovsek@ijs.si
#

from collections import defaultdict
from math import log
import string, itertools
import multiprocessing

import Orange

def chunks(l, n):
    """ Yield n successive chunks from l.
    """
    newn = int(1.0 * len(l) / n + 0.5)
    for i in range(0, n - 1):
        yield l[i * newn:i * newn + newn]
    yield l[n * newn - newn:]


def wordify_examples(name_to_table, connecting_tables, context, index_by_value, target_table_name, word_att_length,
                     ex_idxs):
    cached_sentences = defaultdict(dict)
    wordified_examples = []
    for ex in name_to_table[target_table_name][ex_idxs]:
        wordified_ex = wordify_example(name_to_table, connecting_tables, context, cached_sentences, index_by_value,
                                       target_table_name, word_att_length, target_table_name, ex, set([]))
        wordified_examples.append(wordified_ex)
    return wordified_examples


def wordify_example(name_to_table, connecting_tables, context, cached_sentences, index_by_value, target_table_name,
                    word_att_length, data_name, ex, searched_connections):
    """
    Recursively constructs the 'wordification' document for the given example.

        :param data: The given examples ExampleTable
        :param ex: Example for which the document is constructed
    """
    debug = False
    data_name = str(data_name)

    if debug:
        print("======================================")
        print("example:", ex)
        print("table name:", data_name)
        print("searched_connections:", len(searched_connections), searched_connections)
        print("connecting_tables:", len(connecting_tables[data_name]), connecting_tables[data_name])

    ex_pkey_value = data_name in context.pkeys and ex[str(context.pkeys[data_name])]

    if not data_name in cached_sentences or not str(ex_pkey_value) in cached_sentences[data_name]:

        words = []  # word list for every example
        if debug:
            print("words:", len(words))
        # Construct words (tableName_attributeName_attributeValue) from the given table
        for att in name_to_table[data_name].domain.attributes:
            if not str(att.name) in context.pkeys[data_name] and not str(att.name) in context.fkeys[data_name]:
                words.append(att_to_s(data_name) + "_" + att_to_s(att.name) + "_" + att_to_s(ex[att]))

        # Words from pairs of attributes
        single_words = words[:]
        for comb_length in range(word_att_length + 1):
            if comb_length > 1:
                words.extend(["__".join(sorted(b)) for b in itertools.combinations(single_words, comb_length)])

        # Apply the wordification methodology recursively on all connecting tables
        for sec_t_name, sec_fkey, prim_fkey in connecting_tables[data_name]:
            sec_t = name_to_table[sec_t_name]
            if debug:
                print("------------------")
                print("(sec_t,sec_fkey,prim):", (sec_t_name, sec_fkey, prim_fkey))
                print("search this table:", not (sec_t_name,
                                                 sec_fkey) in searched_connections and sec_t_name != target_table_name)
                print("search this table:", not prim_fkey or not (data_name,
                                                                  sec_fkey) in searched_connections)  # and sec_t!=self.target_table
            if not (sec_t_name, sec_fkey) in searched_connections and sec_t_name != target_table_name and (
                        not prim_fkey or not (data_name, sec_fkey) in searched_connections):
                example_indexes = index_by_value[sec_t_name][str(sec_fkey)][str(ex_pkey_value)] if not prim_fkey else \
                    index_by_value[sec_t_name][str(prim_fkey)][str(ex[str(sec_fkey)])]

                for sec_ex_idx in example_indexes:
                    words += wordify_example(name_to_table, connecting_tables, context, cached_sentences,
                                             index_by_value, target_table_name, word_att_length, sec_t_name,
                                             sec_t[sec_ex_idx], searched_connections | set(
                            [(sec_t_name, sec_fkey), prim_fkey and (data_name, prim_fkey)]))

        cached_sentences[data_name][str(ex_pkey_value)] = words

    return cached_sentences[data_name][str(ex_pkey_value)]


def att_to_s(att):
    """
    Constructs a "wordification" word for the given attribute

        :param att: Orange attribute
        :rtype: str
    """
    return str(att).title().replace(' ', '').replace('_', '')


class Wordification(object):
    def __init__(self, target_table, other_tables, context, word_att_length=1, idf=None):
        """
        Wordification object constructor.

            :param target_table: Orange ExampleTable, representing the primary table
            :param other_tables: secondary tables, Orange ExampleTables
        """
        self.target_table = target_table
        self.other_tables = other_tables
        self.context = context
        self.word_att_length = word_att_length
        self.idf = idf

        self.connecting_tables = defaultdict(list)
        self.cached_sentences = defaultdict(dict)
        self.resulting_documents = []
        self.resulting_classes = []

        self.word_in_how_many_documents = defaultdict(int)
        self.tf_idfs = defaultdict(dict)
        self.name_to_table = {}

        # Finds table connections
        for primary_table in [target_table] + other_tables:
            self.name_to_table[primary_table.name] = primary_table
            for secondary_table in [target_table] + other_tables:
                if (primary_table.name, secondary_table.name) in self.context.connected:
                    for primary_key, foreign_key in self.context.connected[(primary_table.name, secondary_table.name)]:
                        if self.context.pkeys[primary_table.name] == primary_key:
                            self.connecting_tables[primary_table.name].append((secondary_table.name, foreign_key, None))

        self.index_by_value = {}
        for table in [target_table] + other_tables:
            self.index_by_value[table.name] = {}
        for sec_t_name, sec_fkey, prim_fkey in [item for sublist in self.connecting_tables.values() for item in
                                                sublist]:
            sec_t = self.name_to_table[sec_t_name]
            if not prim_fkey:
                self.index_by_value[sec_t.name][sec_fkey] = defaultdict(list)
                for i, ex in enumerate(sec_t):
                    self.index_by_value[sec_t.name][sec_fkey][str(ex[str(sec_fkey)])].append(i)
            else:
                if not prim_fkey in self.index_by_value[sec_t.name]:
                    self.index_by_value[sec_t.name][prim_fkey] = defaultdict(list)

                for i, ex in enumerate(sec_t):
                    self.index_by_value[sec_t.name][prim_fkey][str(ex[str(prim_fkey)])].append(i)

    def run(self, num_of_processes=multiprocessing.cpu_count()):
        """
        Applies the wordification methodology on the target table

            :param num_of_processes: number of processes
        """

        # class + wordification on every example of the main table

        p = multiprocessing.Pool(num_of_processes)

        indices = chunks(list(range(len(self.target_table))), num_of_processes)  # )

        for ex_idxs in indices:
            self.resulting_documents.extend(wordify_examples(self.name_to_table, self.connecting_tables, self.context,
                                                             self.index_by_value, self.target_table.name,
                                                             self.word_att_length, ex_idxs))
        p.close()
        p.join()

        for i, ex in enumerate(self.target_table):
            self.resulting_classes.append(ex.get_class())

    def calculate_weights(self, measure='tfidf'):
        """
        Counts word frequency and calculates tf-idf values for words in every document.

            :param measure: example weights approach (can be one of ``tfidf, binary, tf``).
        """
        # from math import log

        # TODO replace with scipy matrices (and calculate with scikit)

        if measure == 'tfidf':
            self.calculate_idf()

        for doc_idx, document in enumerate(self.resulting_documents):
            train_word_count = defaultdict(int)
            self.tf_idfs[doc_idx] = {}
            for word in document:
                train_word_count[word] += 1

            for word in document:
                if measure == "binary":
                    tf = 1
                    idf = 1
                else:
                    tf = train_word_count[word]
                    idf = 1 if measure == "tf" else (self.idf[word] if word in self.idf else None)

                if idf is not None:
                    self.tf_idfs[doc_idx][word] = tf * idf

    def calculate_idf(self):
        if self.idf:
            return self.idf
        elif len(self.word_in_how_many_documents) != 0:
            raise Exception('Words in document occurence already calculated!')
        else:
            for document in self.resulting_documents:
                for word in set(document):
                    self.word_in_how_many_documents[word] += 1

            no_of_documents = len(self.resulting_documents)
            self.idf = {}
            for word, count in self.word_in_how_many_documents.items():
                self.idf[word] = log(no_of_documents / float(self.word_in_how_many_documents[word]))

    def __check_weights(self):
        if self.tf_idfs == defaultdict(dict):
            raise ValueError('Weights are not available. Call calculate_weights() first.')

    def to_arff(self):
        '''
        Returns the "wordified" representation in ARFF.

            :rtype: str
        '''
        self.__check_weights()

        arff_string = "@RELATION " + self.target_table.name + "\n\n"
        words = set()
        for document in self.resulting_documents:
            for word in document:
                words.add(word)
        words = sorted(words)

        for i, word in enumerate(words):
            arff_string += "@ATTRIBUTE '" + word.replace("'", "") + "' REAL\n"

        arff_string += "@ATTRIBUTE class {" + ','.join(set([str(a) for a in self.resulting_classes])) + "}\n\n@DATA\n"

        self.word_features = []
        for doc_idx in range(len(self.resulting_documents)):
            features = []
            for word in words:
                if word not in self.word_features:
                    self.word_features.append(word)
                if word in self.tf_idfs[doc_idx]:
                    features.append(str(self.tf_idfs[doc_idx][word]))
                else:
                    features.append("0")
            features.append(str(self.resulting_classes[doc_idx]))

            arff_string += ','.join(features)
            arff_string += "\n"

        return arff_string

    def to_orange(self):
        '''
        Returns an Orange Table instance populated with the data.

            :rtype: Orange.data.Table
        '''
        self.__check_weights()

        targetvals = set([a.value for a in self.resulting_classes])
        targetvar = Orange.data.DiscreteVariable(name=self.context.target_att, values=targetvals)

        words = set()
        for document in self.resulting_documents:
            for word in document:
                words.add(word)
        words = sorted(words)
        variables = [Orange.data.ContinuousVariable(name=word) for word in words]
        domain = Orange.data.Domain(variables, class_vars=targetvar)

        data = []
        for doc_idx in range(len(self.resulting_documents)):
            features = []
            for word in words:
                if word in self.tf_idfs[doc_idx]:
                    features.append(self.tf_idfs[doc_idx][word])
                else:
                    features.append(0)
            features.append(self.resulting_classes[doc_idx].value)
            data.append(features)
        table = Orange.data.Table.from_list(domain=domain, rows=data)
        return table

    def prune(self, minimum_word_frequency_percentage=1):
        """
        Filter out words that occur less than minimum_word_frequency times.

            :param minimum_word_frequency_percentage: minimum frequency of words to keep
        """
        pruned_resulting_documents = []

        for document in self.resulting_documents:
            new_document = []
            for word in document:
                if self.word_in_how_many_documents[word] >= minimum_word_frequency_percentage / 100. * len(
                        self.resulting_documents):
                    new_document.append(word)
            pruned_resulting_documents.append(new_document)
        self.resulting_documents = pruned_resulting_documents

    def wordify(self):
        """
        Constructs string of all documents.

            :return: document representation of the dataset, one line per document
            :rtype: str
        """
        string_documents = []
        for klass, document in zip(self.resulting_classes, self.resulting_documents):
            string_documents.append("!" + str(klass) + " " + '' .join(document))
        return '\n'.join(string_documents)

    def att_to_s(self, att):
        """
        Constructs a "wordification" word for the given attribute

            :param att: Orange attribute
        """
        return str(att).title().replace(' ', '').replace('_', '')
