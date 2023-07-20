from logic import *

# Create new classes, each having a name, or a symbol, representing each proposition.
rain = Symbol("rain")  # It is raining.
hagrid = Symbol("hagrid")  # Harry visited Hagrid
dumbledore = Symbol("dumbledore")  # Harry visited Dumbledore

# Save sentences into the KB
knowledge = And(  # Starting from the "And" logical connective, becasue each proposition represents knowledge that we know to be true.

    # 不下雨就去拜访 Hagrid
    Implication(Not(rain), hagrid),  # ¬(It is raining) → (Harry visited Hagrid)

    # 或者 两个都有可能被拜访: 两者的并集
    Or(hagrid, dumbledore),  # (Harry visited Hagrid) ∨ (Harry visited Dumbledore).

    # 不可能两个都去拜访: 两者的交集
    Not(And(hagrid, dumbledore)),  # ¬(Harry visited Hagrid ∧ Harry visited Dumbledore) i.e. Harry did not visit both Hagrid and Dumbledore.

    # 哈利去找邓布利多 
    dumbledore  # Harry visited Dumbledore. Note that while previous propositions contained multiple symbols with connectors, this is a proposition consisting of one symbol. This means that we take as a fact that, in this KB, Harry visited Dumbledore.
    )

print(model_check(knowledge, rain)) # 根据构建出来的知识工程来判断 "rain" 状态的真假：也就是通过上面的情况来推断下雨的真假