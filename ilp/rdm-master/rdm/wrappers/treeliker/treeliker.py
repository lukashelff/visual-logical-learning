import shutil
import tempfile
import os.path
from subprocess import Popen, PIPE


class TreeLiker:
    '''
    TreeLiker python wrapper.
    '''
    def __init__(self, dataset, template, test_dataset=None, settings={}):
        '''
            :param dataset: dataset in TreeLiker format
            :param template: feature template
            :param test_dataset: (optional) test dataset to transform with the features from the training set
            :param settings: dictionary of settings (see `TreeLiker documentation <http://ida.felk.cvut.cz/treeliker/download/treeliker.pdf>`_)
        '''
        self.basename = 'default'
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.template = template
        self.settings = settings

    def _copy_data(self):
        self.tmpdir = tempfile.mkdtemp()

        with open('%s/%s.txt' % (self.tmpdir, self.basename), 'w') as f:
            f.write(self.dataset)

        if self.test_dataset:
            with open('%s/%s_test.txt' % (self.tmpdir, self.basename), 'w') as f:
                f.write(self.test_dataset)

        # Copy binaries to tmp folder
        cdir = os.path.dirname(os.path.abspath(__file__))
        shutil.copytree('%s/bin/' % cdir, '%s/bin/' % self.tmpdir)

    def run(self, cleanup=True, printOutput=False):
        '''
        Runs TreeLiker with the given settings.

            :param cleanup: deletes temporary files after completion
            :param printOutput: print algorithm output to the terminal
        '''
        self._copy_data()
        self._batch()

        dumpFile = None
        if not printOutput:
            dumpFile = tempfile.TemporaryFile()

        p = Popen(['java', '-Xmx3G', '-cp', 'bin/TreeLiker.jar', 
                   'ida.ilp.treeLiker.TreeLikerMain', '-batch', self.batch], 
                   cwd=self.tmpdir,
                   stdout=dumpFile,
                   stderr=dumpFile)
        stdout_str, stderr_str = p.communicate()

        if not self.test_dataset:
            arff = open('%s/%s.arff' % (self.tmpdir, self.basename)).read()
            arff_test = None
        else:
            arff = open('%s/conversion/train.arff' % self.tmpdir).read()
            arff_test = open('%s/conversion/test.arff' % self.tmpdir).read()

        if cleanup:
            self._cleanup()
        
        return (arff, arff_test)

    def _batch(self):
        '''
        Creates the batch file to run the experiment.
        '''
        self.batch = '%s/%s.treeliker' % (self.tmpdir, self.basename)

        commands = []
        if not self.test_dataset:
            commands.append('set(output_type, single)')
            commands.append("set(examples, '%s.txt')" % self.basename)
        else:
            commands.append('set(output_type, train_test)')
            commands.append("set(train_set, '%s.txt')" % self.basename)
            commands.append("set(test_set, '%s_test.txt')" % self.basename)
        commands.append('set(template, %s)' % self.template)

        if not self.test_dataset:
            commands.append('set(output, %s.arff)' % self.basename)
        else:
            commands.append('set(output, conversion)')

        # Optional settings
        for key, val in self.settings.items():
            if val not in [None, '']:
                commands.append('set(%s, %s)' % (key, str(val)))

        commands.append('work(yes)')

        with open(self.batch, 'w') as f:
            f.write('\n'.join(commands))

    def _cleanup(self):
        """
        Cleans up all the temporary files.
        """
        try:
            shutil.rmtree(self.tmpdir)
        except:
            logger.info('Problem removing temporary files. \
                         The files are probably in use.')
