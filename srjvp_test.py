#%% Imports
from adjax import *
from rich import inspect


#%%
# add primatives for semiring operations

semiring_add_p = Primitive('semiring_add')
semiring_mul_p = Primitive('semiring_mul')

def semiring_add(x, y): return bind1(semiring_add_p, x, y)
def semiring_mul(x, y): return bind1(semiring_mul_p, x, y)

# temp just define these as| normal add and mul
impl_rules[semiring_add_p] = lambda x, y: x + y
impl_rules[semiring_mul_p] = lambda x, y: x * y

abstract_eval_rules[semiring_add_p] = binop_abstract_eval
abstract_eval_rules[semiring_mul_p] = binop_abstract_eval

#%%
# add some extra primatives needed here

exp_p = Primitive('exp')

def exp(x): return bind1(exp_p, x)

def exp_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return [ exp(x) ], [ exp(x) * x_dot ]

def exp_srjvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return [ exp(x) ], [ semiring_mul(exp(x), x_dot) ]

impl_rules[exp_p] = lambda x: np.exp(x)
jvp_rules[exp_p] = exp_jvp
abstract_eval_rules[exp_p] = vectorized_unop_abstract_eval

#%%
# one approach is to use a tracer that carries around the tangent and the semiring operations
# with the expectation that 

# every binary operation 

class SRJVPTracer(Tracer):
    def __init__(self, trace, primal, tangent):
        # print(f"new srjvp tracer {trace} {primal} {tangent}")
        self._trace = trace
        self.primal = primal
        self.tangent = tangent
    
    @property
    def aval(self):
        return get_aval(self.primal)
    
class SRJVPTrace(Trace):
    pure = lift = lambda self, val: SRJVPTracer(self, val, zeros_like(val))

    def process_primitive(self, primitive, tracers, params):
        # print(f"SRJVPTrace: process_primitive {primitive} {tracers} {params}")
        primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
        srjvp_rule = srjvp_rules[primitive]
        primal_outs, tangents_outs = srjvp_rule(primals_in, tangents_in, **params)
        return [SRJVPTracer(self, p, t) for p, t in zip(primal_outs, tangents_outs)]
    
srjvp_rules = {}

#%%
# Primitive JVP rules for semiring operations

def add_srjvp(primals, tangents):
  (x, y), (x_dot, y_dot) = primals, tangents                                                                                                                                                                                                                            
  return [x + y], [semiring_add(x_dot , y_dot)]

def mul_srjvp(primals, tangents):
  (x, y), (x_dot, y_dot) = primals, tangents
  return [x * y], [semiring_add(semiring_mul(x_dot , y) , semiring_mul(x , y_dot))]

def sin_srjvp(primals, tangents):
  (x,), (x_dot,) = primals, tangents
  return [sin(x)], [semiring_mul(cos(x) , x_dot)]

def cos_srjvp(primals, tangents):
  (x,), (x_dot,) = primals, tangents
  return [cos(x)], [semiring_mul(-sin(x) , x_dot)]

# not sure how to handle these 
def neg_srjvp(primals, tangents):                                                                      
  (x,), (x_dot,) = primals, tangents
  return [neg(x)], [neg(x_dot)]

def reduce_sum_srjvp(primals, tangents, *, axis):
  (x,), (x_dot,) = primals, tangents
  return [reduce_sum(x, axis)], [reduce_sum(x_dot, axis)]

def greater_srjvp(primals, tangents):
  (x, y), _ = primals, tangents
  out_primal = greater(x, y)
  return [out_primal], [zeros_like(out_primal)]

def less_srjvp(primals, tangents):
  (x, y), _ = primals, tangents
  out_primal = less(x, y)
  return [out_primal], [zeros_like(out_primal)]

#%%
# table of rules for srjvp

srjvp_rules[add_p] = add_srjvp
srjvp_rules[mul_p] = mul_srjvp
srjvp_rules[sin_p] = sin_srjvp
srjvp_rules[cos_p] = cos_srjvp
srjvp_rules[neg_p] = neg_srjvp
srjvp_rules[reduce_sum_p] = reduce_sum_srjvp
srjvp_rules[greater_p] = greater_srjvp
srjvp_rules[less_p] = less_srjvp
srjvp_rules[exp_p] = exp_srjvp

#%%
def srjvp_flat(f, primals, tangents):
    # print("srjvp_flat enter")
    with new_main(SRJVPTrace) as main:
        # print("srjvp context manager entry")
        trace = SRJVPTrace(main)
        # print("main SRJVP trace created")
        tracers_in = [SRJVPTracer(trace, x, t) for x, t in zip(primals, tangents)]
        outs = f(*tracers_in)
        tracers_out = [full_raise(trace, out) for out in outs]
        primals_out, tangents_out = unzip2((t.primal, t.tangent) for t in tracers_out)
    return primals_out, tangents_out

def srjvp(f, primals, tangents):
    # print(f"srjvp {primals} {tangents}")
    primals_flat, in_tree = tree_flatten(primals)
    tangents_flat, in_tree2 = tree_flatten(tangents)
    if in_tree != in_tree2:
        raise TypeError
    f, out_tree = flatten_fun(f, in_tree)
    primals_out_flat, tangents_out_flat = srjvp_flat(f, primals_flat, tangents_flat)
    primals_out = tree_unflatten(out_tree(), primals_out_flat)
    tangents_out = tree_unflatten(out_tree(), tangents_out_flat)
    return primals_out, tangents_out

def srjvp_jaxpr(jaxpr: Jaxpr) -> tuple[Jaxpr, list[Any]]:
    def srjvp_traceable(*primals_and_tangents):
        n = len(primals_and_tangents) // 2
        primals, tangents = primals_and_tangents[:n], primals_and_tangents[n:]
        return srjvp(jaxpr_as_fun(jaxpr), primals, tangents)
    
    in_avals = [v.aval for v in jaxpr.in_binders]
    new_jaxpr, new_consts, _ = make_jaxpr(srjvp_traceable, *in_avals, *in_avals)
    return new_jaxpr, new_consts

#%%
# Test function from semiring backprop paper

def f(x,y):
    return exp(x) + (x + -y)*y

jaxpr, consts, _ = make_jaxpr(f, get_aval(1.0), get_aval(1.0))
print(jaxpr)

jaxpr_jvp, consts_jvp = jvp_jaxpr(jaxpr)
print(jaxpr_jvp)



#%%
# Comparison of jvp and srjvp as well as srjvp with partial eval on the test function

def f(x,y):
    return exp(x) + (x + -y)*y

jaxpr, consts, _ = make_jaxpr(f, get_aval(1.0), get_aval(1.0))

jaxpr_jvp, consts_jvp = jvp_jaxpr(jaxpr)

jaxpr_srjvp, consts_srjvp = srjvp_jaxpr(jaxpr)

print("jaxpr - raw expression")
print(jaxpr)

print("jaxpr - jvp")
print(jaxpr_jvp)

print("jaxpr - srjvp")
print(jaxpr_srjvp)

#%%
# partial eval

in_unknowns = [True, False]
jaxpr1, jaxpr2, out_unknowns, num_res = partial_eval_jaxpr(jaxpr_srjvp, in_unknowns)

print("jaxpr1")

print(jaxpr1)

print("jaxpr2")

print(jaxpr2)




#%%
#%%
# Comparison of jvp and srjvp as well as srjvp with partial eval on the test function

def f(x):
    return 3.0 * x + 2.0

jaxpr, consts, _ = make_jaxpr(f, get_aval(1.0))

jaxpr_jvp, consts_jvp = jvp_jaxpr(jaxpr)

jaxpr_srjvp, consts_srjvp = srjvp_jaxpr(jaxpr)

print("jaxpr - raw expression")
print(jaxpr)

print("jaxpr - jvp")
print(jaxpr_jvp)

print("jaxpr - srjvp")
print(jaxpr_srjvp)

#%%
def srlinearize_flat(f, *primals_in):
    print("linearize flat")

    pvals_in = ([PartialVal.known(x) for x in primals_in] +
                [PartialVal.unknown(vspace(get_aval(x))) for x in primals_in])
    
    # print(pvals_in)

    def f_srjvp(*primals_tangents_in):
        primals_out, tangents_out = srjvp(f, *split_half(primals_tangents_in))
        return [*primals_out, *tangents_out]
    
    jaxpr, pvals_out, consts = partial_eval_flat(f_srjvp, pvals_in)
    primal_pvals, _ = split_half(pvals_out)
    assert all(pval.is_known for pval in primal_pvals)
    primals_out = [pval.const for pval in primal_pvals]

    print(jaxpr)

    f_lin = lambda *tangents: eval_jaxpr(jaxpr, [*consts, *tangents])

    return primals_out, f_lin, jaxpr

def srlinearize(f, *primals_in):
    print("srlinearize")
    primals_in_flat, in_tree = tree_flatten(primals_in)
    f, out_tree = flatten_fun(f, in_tree)
    primals_out_flat, f_lin_flat, jaxpr = srlinearize_flat(f, *primals_in_flat)
    primals_out = tree_unflatten(out_tree(), primals_out_flat)

    def f_lin(*tangents_in):
        tangents_in_flat, in_tree2 = tree_flatten(tangents_in)
        if in_tree != in_tree2: raise TypeError
        tangents_out_flat = f_lin_flat(*tangents_in_flat)
        return tree_unflatten(out_tree(), tangents_out_flat)

    return primals_out, f_lin, jaxpr

def f(x):
    return 2.0*x + 1.0

x, f_lin, jaxpr = srlinearize(f, 1.0)
print(jaxpr)
# %%
