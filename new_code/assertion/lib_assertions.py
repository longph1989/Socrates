import enum


# class Assertion:
#     def __init__(self, impls):
#         self.impls = impls
#
#     def get_value(self, vars_dict):
#         for impl in self.impls:
#             if impl.get_value(vars_dict):
#                 return True
#         return False


class Implication:
    def __init__(self, vars, pre, post, init_dict):
        self.vars = vars
        self.pre = pre
        self.post = post
        self.init_dict = init_dict

    def get_value(self, vars_dict):
        vars_dict.update(self.init_dict)

        if not self.get_pre_value(vars_dict):
            return True
        elif self.get_post_value(vars_dict):
            return True
        else:
            return False

    def get_pre_value(self, vars_dict):
        vars_dict.update(self.init_dict)
        return pre.get_value(vars_dict)

    def get_post_value(self, vars_dict):
        vars_dict.update(self.init_dict)
        return post.get_value(vars_dict)


class Disjunction:
    def __init__(self, conjs):
        self.conjs = conjs

    def get_value(self, vars_dict):
        for conj in self.conjs:
            if conj.get_value(vars_dict):
                return True
        return False


class Conjunction:
    def __init__(self, terms):
        self.terms = terms

    def get_value(self, vars_dict):
        for term in terms:
            if not term.get_value(vars_dict):
                return False
        return True


class TrueTerm:
    def __init__(self):
        pass

    def get_value(self, vars_dict):
        return True


class GeneralTerm:
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def get_value(self, vars_dict):
        lhs_value = lhs.get_value(vars_dict)
        rhs_value = rhs.get_value(vars_dict)

        if self.op == Op.GE:
            return True if lhs_value >= rhs_value else False
        elif self.op == Op.GT:
            return True if lhs_value > rhs_value else False
        elif self.op == Op.LE:
            return True if lhs_value <= rhs_value else False
        elif self.op == Op.LT:
            return True if lhs_value < rhs_value else False
        elif self.op == Op.EQ:
            return True if lhs_value == rhs_value else False
        elif self.op == Op.NE:
            return True if lhs_value != rhs_value else False


class Function:
    def __init__(self, func, vars):
        self.func = func
        self.vars = vars

    def get_value(self, vars_dict):
        args = list()

        for var in self.vars:
            args.append(vars_dict[var.name])

        return func(*args)


class Var:
    def __init__(self, name):
        self.name = name


class Op(enum.Enum):
   GE = 1
   GT = 2
   LE = 3
   LT = 4
   EQ = 5
   NE = 6
