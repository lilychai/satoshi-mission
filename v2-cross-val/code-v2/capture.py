## Code from: http://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call

from cStringIO import StringIO
import sys

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout


##--- Usage ---
# with Capturing() as output:
#     print 'hello world'
#
# print 'displays on screen'
#
# with Capturing(output) as output:  # note the constructor argument
#     print 'hello world2'
#
# print 'done'
# print 'output:', output

##--- Output ---
# displays on screen
# done
# output: ['hello world', 'hello world2']
