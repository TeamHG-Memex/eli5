import jinja2
import sklearn


def test_import():
	from packaging import version
	if version.parse(jinja2.__version__) <= version.parse('3.0.3'):
		raise ImportError('module "Jinja2" must be of a later version than "3.0.3".')
	if version.parse(sklearn.__version__) < version.parse('1.0.2'):
		raise ImportError('module "Jinja2" must be version "1.0.2" or later.')
	import eli5


if __name__ == '__main__':
	test_import()
