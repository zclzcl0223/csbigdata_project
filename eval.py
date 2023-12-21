from ogb.nodeproppred import Evaluator

"""
evaluator = Evaluator(name = 'ogbn-products')
print(evaluator.expected_input_format) 
print(evaluator.expected_output_format)
result_dict = evaluator.eval(input_dict)
"""

def evaluate(y_pred, y_true):

    evaluator = Evaluator(name = 'ogbn-products')
    input_dict = {}
    input_dict['y_true'] = y_true.reshape(-1, 1)
    input_dict['y_pred'] = y_pred.max(dim=1)[1].reshape(-1, 1)
    
    return evaluator.eval(input_dict)['acc']