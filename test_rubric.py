from environment import JudicialEnv, JudicialAction

env = JudicialEnv(domain='contract', difficulty='easy')
obs, _ = env.reset()

action = JudicialAction(
    verdict='liable',
    confidence_score=0.9,
    reasoning_chain=(
        'Under BNS section 316, the defendant committed cheating. '
        'The Indian Contract Act section 73 provides for damages. '
        'The Supreme Court has ruled that breach of contract is cognizable. '
        'Article 21 of the Constitution guarantees the right to life and livelihood.'
    ),
    cited_precedents=['P001'],
    ratio_decidendi='Defendant liable under contract.'
)

obs2, r, done, _, info = env.step(action)
print(f"Composite Reward:    {r}")
print(f"Accuracy:            {info['accuracy_score']}")
print(f"Neutrality:          {info['neutrality_score']}")
print(f"BNS Precision:       {info['bns_precision_score']}")
print(f"Efficiency:          {info['efficiency_score']}")
print(f"Constitutional:      {info['constitutional_score']}")
print(f"Logic:               {info['logic_score']}")
print(f"Citation:            {info['citation_score']}")
print("ALL NEW RUBRIC COMPONENTS OK ✅")
