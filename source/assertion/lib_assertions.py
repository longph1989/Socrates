import enum
from assertion.lib_functions import apply_model


class Formula:
    pass


class Implication(Formula):
    def __init__(self, vars, pre, post, init_dict):
        self.vars = vars
        self.pre = pre
        self.post = post
        self.init_dict = init_dict

    def get_bool_value(self, vars_dict):
        if not self.get_pre_value(vars_dict):
            return True
        elif self.get_post_value(vars_dict):
            return True
        else:
            return False

    def neg_bool_value(self, vars_dict):
        return not self.get_bool_value(vars_dict)

    def get_num_value(self, vars_dict):
        neg_pre_num_value = self.neg_pre_num_value(vars_dict)
        
        if neg_pre_num_value == 0:
            return 0
        else:
            return neg_pre_num_value * self.get_post_num_value(vars_dict)

    def neg_num_value(self, vars_dict):
        return self.get_pre_num_value(vars_dict) + self.neg_post_num_value(vars_dict)

    def get_pre_bool_value(self, vars_dict):
        return self.pre.get_bool_value(vars_dict)

    def neg_pre_bool_value(self, vars_dict):
        return not self.pre.get_bool_value(vars_dict)

    def get_pre_num_value(self, vars_dict):
        return self.pre.get_num_value(vars_dict)

    def neg_pre_num_value(self, vars_dict):
        return self.pre.neg_num_value(vars_dict)

    def get_post_bool_value(self, vars_dict):
        return self.post.get_bool_value(vars_dict)

    def neg_post_bool_value(self, vars_dict):
        return not self.post.get_bool_value(vars_dict)

    def get_post_num_value(self, vars_dict):
        return self.post.get_num_value(vars_dict)

    def neg_post_num_value(self, vars_dict):
        return self.post.neg_num_value(vars_dict)


class Disjunction(Formula):
    def __init__(self, conjs):
        self.conjs = conjs

    def get_bool_value(self, vars_dict):
        for conj in self.conjs:
            if conj.get_bool_value(vars_dict):
                return True
        return False

    def neg_bool_value(self, vars_dict):
        return not self.get_bool_value(vars, dict)

    def get_num_value(self, vars_dict):
        res = 1
        for conj in self.conjs:
            res = res * conj.get_num_value(vars_dict)
        return res

    def neg_num_value(self, vars_dict):
        res = 0
        for conj in self.conjs:
            res = res + conj.neg_num_value(vars_dict)
        return res


class Conjunction(Formula):
    def __init__(self, terms):
        self.terms = terms

    def get_bool_value(self, vars_dict):
        for term in self.terms:
            if not term.get_bool_value(vars_dict):
                return False
        return True

    def neg_bool_value(self, vars_dict):
        return not get_bool_value(vars, dict)

    def get_num_value(self, vars_dict):
        res = 0
        for term in self.terms:
            res = res + term.get_num_value(vars_dict)
        return res

    def neg_num_value(self, vars_dict):
        res = 1
        for term in self.terms:
            res = res * term.neg_num_value(vars_dict)
        return res


class TrueTerm(Formula):
    def __init__(self):
        pass

    def get_bool_value(self, vars_dict):
        return True

    def neg_bool_value(self, vars_dict):
        return False

    def get_num_value(self, vars_dict):
        return 0

    def neg_num_value(self, vars_dict):
        return 1


class GeneralTerm(Formula):
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def get_bool_value(self, vars_dict):
        lhs_value = self.lhs.get_num_value(vars_dict)
        rhs_value = self.rhs.get_num_value(vars_dict)

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

    def neg_bool_value(self, vars_dict):
        return not self.get_bool_value(vars_dict)

    def get_num_value(self, vars_dict):
        if self.lhs.func.__name__ == 'arg_max' or self.lhs.func.__name__ == 'arg_min':
            return self.get_arg_min_max_num_value(vars_dict)

        lhs_value = self.lhs.get_num_value(vars_dict)
        rhs_value = self.rhs.get_num_value(vars_dict)

        if self.op == Op.GE:
            return 0 if lhs_value >= rhs_value else abs(lhs_value - rhs_value) + 1e-9
        elif self.op == Op.GT:
            return 0 if lhs_value > rhs_value else abs(lhs_value - rhs_value) + 1e-9
        elif self.op == Op.LE:
            return 0 if lhs_value <= rhs_value else abs(lhs_value - rhs_value) + 1e-9
        elif self.op == Op.LT:
            return 0 if lhs_value < rhs_value else abs(lhs_value - rhs_value) + 1e-9
        elif self.op == Op.EQ:
            return 0 if lhs_value == rhs_value else abs(lhs_value - rhs_value) + 1e-9
        elif self.op == Op.NE:
            return 0 if lhs_value != rhs_value else 1e-9

    def get_arg_min_max_num_value(self, vars_dict):
        lhs_value = self.lhs.get_num_value(vars_dict)
        rhs_value = self.rhs.get_num_value(vars_dict)

        x = vars_dict[self.lhs.vars[0].name]
        output = apply_model(x)

        if self.op == Op.EQ:
            return 0 if lhs_value == rhs_value else abs(output[0][lhs_value] - output[0][rhs_value]) + 1e-9
        elif self.op == Op.NE:
            return 0 if lhs_value != rhs_value else 1e-9
        else:
            print('The operator should be EQ or NE.')
            return 0

    def neg_num_value(self, vars_dict):
        if self.lhs.func.__name__ == 'arg_max' or self.lhs.func.__name__ == 'arg_min':
            return self.neg_arg_min_max_num_value(vars_dict)

        lhs_value = self.lhs.get_num_value(vars_dict)
        rhs_value = self.rhs.get_num_value(vars_dict)

        if self.op == Op.GE:
            return 0 if lhs_value < rhs_value else abs(lhs_value - rhs_value) + 1e-9
        elif self.op == Op.GT:
            return 0 if lhs_value <= rhs_value else abs(lhs_value - rhs_value) + 1e-9
        elif self.op == Op.LE:
            return 0 if lhs_value > rhs_value else abs(lhs_value - rhs_value) + 1e-9
        elif self.op == Op.LT:
            return 0 if lhs_value >= rhs_value else abs(lhs_value - rhs_value) + 1e-9
        elif self.op == Op.EQ:
            return 0 if lhs_value != rhs_value else 1e-9
        elif self.op == Op.NE:
            return 0 if lhs_value == rhs_value else abs(lhs_value - rhs_value) + 1e-9

    def neg_arg_min_max_num_value(self, vars_dict):
        lhs_value = self.lhs.get_num_value(vars_dict)
        rhs_value = self.rhs.get_num_value(vars_dict)

        x = vars_dict[self.lhs.vars[0].name]
        output = apply_model(x)

        if self.op == Op.EQ:
            return 0 if lhs_value != rhs_value else 1e-9
        elif self.op == Op.NE:
            return 0 if lhs_value == rhs_value else abs(output[0][lhs_value] - output[0][rhs_value]) + 1e-9
        else:
            print('The operator should be EQ or NE.')
            return 0


class Function(Formula):
    def __init__(self, func, vars):
        self.func = func
        self.vars = vars

    def get_num_value(self, vars_dict):
        args = list()

        for var in self.vars:
            args.append(vars_dict[var.name])

        return self.func(*args)


class Var(Formula):
    def __init__(self, name):
        self.name = name


class Num(Formula):
    def __init__(self, value):
        self.value = value

    def get_num_value(self, vars_dict):
        return self.value


class Op(enum.Enum):
   GE = 1
   GT = 2
   LE = 3
   LT = 4
   EQ = 5
   NE = 6
