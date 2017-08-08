#!/usr/bin/env python
# -*- coding: utf-8 -*-

# flake8: noqa

import six

if six.PY3:  # pragma: no cover
    # from: http://code.activestate.com/recipes/578587/
    # license: MIT

    from functools import partial

    from .mro import mro

    # This definition is only used to assist static code analyzers
    def doc_inherit(fn):
        '''Copy docstring for method from superclass

        For this decorator to work, the class has to use the
        `InheritableDocstrings` metaclass.
        '''
        raise RuntimeError('Decorator can only be used in classes '
                           'using the `InheritableDocstrings` metaclass')

    def _doc_inherit(mro, fn):
        '''Decorator to set docstring for *fn* from *mro*'''

        if fn.__doc__ is not None:
            raise RuntimeError('Function already has docstring')

        # Search for docstring in superclass
        for cls in mro:
            super_fn = getattr(cls, fn.__name__, None)
            if super_fn is None:
                continue
            fn.__doc__ = super_fn.__doc__
            break
        else:
            raise RuntimeError(
                "Can't inherit docstring for %s: method does not "
                "exist in superclass" % fn.__name__)

        return fn

    class InheritableDocstrings(type):

        @classmethod
        def __prepare__(cls, name, bases, **kwds):
            classdict = super().__prepare__(name, bases, *kwds)

            # Inject decorators into class namespace
            classdict['doc_inherit'] = partial(_doc_inherit, mro(*bases))

            return classdict

        def __new__(cls, name, bases, classdict):

            # Decorator may not exist in class dict if the class (metaclass
            # instance) was constructed with an explicit call to `type`.
            # (cf http://bugs.python.org/issue18334)
            if 'copy_ancestor_docstring' in classdict:

                # Make sure that class definition hasn't messed with decorators
                copy_impl = getattr(
                    classdict['doc_inherit'], 'func', None)
                if copy_impl is not _copy_ancestor_docstring:
                    raise RuntimeError(
                        'No copy_ancestor_docstring attribute may be created '
                        'in classes using the InheritableDocstrings metaclass')

                # Delete decorators from class namespace
                del classdict['doc_inherit']

            return super().__new__(cls, name, bases, classdict)

else:  # pragma: no cover

    # from: http://code.activestate.com/recipes/576862/
    # license: MIT

    # doc_inherit decorator

    # Usage:

    # class Foo(object):
        # def foo(self):
            # "Frobber"
            # pass

    # class Bar(Foo):
        # @doc_inherit
        # def foo(self):
            # pass

    # Now, Bar.foo.__doc__ == Bar().foo.__doc__ == Foo.foo.__doc__ == "Frobber"

    from functools import wraps

    class DocInherit(object):
        """
        Docstring inheriting method descriptor

        The class itself is also used as a decorator
        """

        def __init__(self, mthd):
            self.mthd = mthd
            self.name = mthd.__name__

        def __get__(self, obj, cls):
            if obj:
                return self.get_with_inst(obj, cls)
            else:
                return self.get_no_inst(cls)

        def get_with_inst(self, obj, cls):

            overridden = getattr(super(cls, obj), self.name, None)

            @wraps(self.mthd, assigned=('__name__','__module__'))
            def f(*args, **kwargs):
                return self.mthd(obj, *args, **kwargs)

            return self.use_parent_doc(f, overridden)

        def get_no_inst(self, cls):

            for parent in cls.__mro__[1:]:
                overridden = getattr(parent, self.name, None)
                if overridden: break

            @wraps(self.mthd, assigned=('__name__','__module__'))
            def f(*args, **kwargs):
                return self.mthd(*args, **kwargs)

            return self.use_parent_doc(f, overridden)

        def use_parent_doc(self, func, source):
            if source is None:
                raise NameError("Can't find '%s' in parents" % self.name)
            func.__doc__ = source.__doc__
            return func

    class InheritableDocstrings(type): pass  # this is for py3 comp
    doc_inherit = DocInherit
