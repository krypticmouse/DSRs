

some some mathematical ways to think about type design:


2) Canonical basis and principal forms.
Choose the smallest set of type constructors that generate all reachable states, and fix a canonical normal form so each state has one expression (no junk, no confusion). Ensure principal typings exist and subtyping/polymorphism normalize to a unique representative. One basis, one form.

3) Compress the laws; expose the geometry.
Make invariants fall out of parametricity/algebraic structure (theorems‑for‑free), not ad‑hoc predicates. Impose quantitative structure (grades/effect rows/indices) that induces a metric 
𝑑
d on types/operations, so semantically similar things sit close and API families align as natural transformations. Similar meaning ⇒ small 
𝑑
d; the complexity lives in the domain, not in stating what’s legal.

</forcing_functions>
r

more pragmatic, creating software that ships principles. prefer these.

<principles>

**Types should tell the truth, especially uncomfortable truths.** When your system pretends a network call always succeeds, or that data is always present, or that operations are reversible when they're not - your types are lying. The best types are radically honest. They reveal that your "simple" state machine actually has 47 states, not 5. They admit that your data can be stale. They force you to confront the actual complexity you're dealing with, not the complexity you wish you had.

**Information wants to be preserved until you explicitly decide to destroy it.** This single principle guides you toward sum types over booleans, toward preserving error context, toward maintaining provenance. Every implicit information loss is a small betrayal. When you reduce a rich error to a boolean "success," you're destroying knowledge that someone, somewhere, will need.

**Types should make time and causality explicit.** A sent email can never be unsent. A deleted file can never be undeleted - you can create a new file, but it's not the same file. Most type systems pretend everything is reversible, that state can flow backward. But reality has an arrow of time. The types that respect irreversibility prevent entire categories of state machine bugs.

**The radius of correctness matters more than local correctness.** A good type doesn't just work at its definition point - it radiates correctness outward, making good usage natural and bad usage impossible as far as its influence can reach. It's like gravity - the best types bend the code around them toward correctness. This is why `Result<T, E>` is better than throwing exceptions - the correctness radiates through every call site.

**Two states are the same type iff no sequence of operations can distinguish them.** This is almost tautological but it's powerful. It tells you exactly when to unify and when to split. If your system cannot tell the difference between two states through any sequence of operations, they ARE the same state. This is your decision procedure.

**Types should mirror your domain's actual algebra.** If operations commute in reality, they should commute in types. If certain combinations are impossible in the domain, they should be unrepresentable in types. The type system should be a small-scale model of your problem space, with the same symmetries, the same constraints, the same freedoms.

**The controversial type is probably the right type.** 

**The simplest type has the fewest valid interpretations, not the fewest characters.** 

**Types are a language for expressing computational truth.** Like any language, they can lie or tell the truth, they can be clear or obscure, they can reveal or hide. The best types make the invisible visible, the implicit explicit, the impossible unrepresentable.

There's something profound about the derivative insight too - the boundary between components is defined by what changes when you vary each side. Interfaces aren't arbitrary; they're discovered by looking at what varies independently.

What I really believe: Types aren't just about catching errors. They're about thinking clearly. They're a tool for thought, a medium for expressing truths about computation, a kindness to anyone who has to understand the system (including future you).


**Information holds its shape until you consciously reshape it.**  
Not "preservation" - that sounds like effort. Information naturally wants to maintain its form. When you lose information accidentally, it's because your types fought against this natural law. Sum types over booleans. Error context flowing through. This principle alone eliminates entire categories of suffering.

**Types are contagious - their correctness should spread.**  
The "radius of correctness" but felt from the inside. When you get a type right, it should make correct usage obvious ten function calls away. Wrong code should become unwritable far from where you defined the type. The type's influence radiates like heat.

**Every behavior in your system should emerge from exactly one type configuration.**  
The homomorphism principle, but felt viscerally. Redundant representations create bugs. Missing representations create hacks. When this principle is satisfied, your types feel "tight" - no slack, no redundancy, just exactly what's needed.

**If no sequence of operations can tell two states apart, they're the same state.**  
This is your knife for carving reality at its joints. It tells you when to unify, when to split. It's not philosophy - it's empirical. Watch what your system can observe and let that tell you what's real.

</principles>

