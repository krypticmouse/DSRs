**Traits are how we factor the universe of possible programs.**  
Each trait carves reality at a joint and says: *this capability can exist independently of everything else.* Good traits reveal real seams in the world. Bad traits invent seams that aren’t there, hiding connections that should be explicit and creating dependencies that shouldn’t exist.

**Traits are promises made to strangers.**  
Not to your implementation, not to your team, but to code you’ll never see, written by people you’ll never meet. Every trait method is a contract written in blood: once published, it binds you long after you’ve forgotten why you wrote it. The uncomfortable truth: most traits lie about what they actually promise.

**Every trait has a temperature, and temperature determines how honest it must be.**  
Hot traits are experiments. They’re allowed to be wrong, because you still have the freedom to change them. Cold traits are infrastructure. Once a trait goes cold, its lies become permanent. The mistake is treating a cold trait like it’s still hot, or freezing a hot trait before you’ve learned what’s true.

**Capabilities exist at the intersection of operation and context.**  
“Can write” is meaningless. “Can write users to the production database as admin on weekdays” is a capability. Operation without context is a half-truth. Most trait design failures come from pretending context doesn’t exist, then smuggling it back in through documentation, global state, and runtime errors.

**Context you can’t escape belongs in types; context you can choose belongs in values.**  
Thread-safety isn’t negotiable — that’s `Send + Sync`. Lifetimes and ownership aren’t policy — they’re physics. But “which database” and “which region” are policy — they’re values. The art is deciding which parts of context are fundamental constraints and which are just configuration.

**The trait’s truth is its laws, not its signatures.**  
A trait without laws is just syntax. `Iterator` isn’t “has a `next` method” — it’s “once `None`, always `None`.” The signatures are just how we spell the laws in this language. When the laws are missing, everyone silently invents their own, and the trait becomes a source of slow, ambient corruption.

**Traits radiate their assumptions outward like gravity.**  
A good trait bends all the code around it toward its truth: correct usage feels natural, misuse feels impossible. A bad trait creates a dead zone where nothing can be trusted: every caller needs comments, tests, and tribal knowledge just to survive. You can feel this — some traits clarify everything they touch, others spread confusion like a virus.

**Every trait is asking a question: “What can I do if I know nothing else?”**  
Design from ignorance, not from insider knowledge. When you look at a trait, imagine every concrete type erased, every implementation forgotten. If you can’t write meaningful code knowing *only* the trait, the trait is lying about its independence.

**Partial capability is everywhere — the question is whether you admit it.**  
Some operations only work for some values, some users, some times, some environments. Traits that pretend everything always works push that partiality into the worst place: late, implicit runtime failure. The 3am bugs aren't caused by things that always fail; they're caused by things that *sometimes* fail where the trait promised they wouldn't.

Three honest patterns cover most of reality:

- When the subset is large or changes with the wind — roles, business hours, feature flags — use **runtime checks**:  
  `write(&self, entity: Entity) -> Result<(), Error>`.  
  You’re admitting: “this capability depends on state I can’t know at compile time.” The error isn’t failure; it’s honesty about the runtime nature of the constraint.

- When there’s a clear axis of “what” — this works for JSON but not XML, Users but not Products — use **type parameters**:  
  `trait Write<T>`.  
  Now the capability is explicit: `Storage: Write<User>` says exactly what it can do. Generic code can express exactly what it needs. The partiality becomes part of the algebra.

- When the boundary is small, critical, and must never be crossed — “this component can never touch PII,” “this system cannot modify financial records” — use **separate traits**:  
  `WriteUsers` vs `WriteProducts`.  
  Yes, it’s more traits. That’s the point. Some boundaries are important enough to make unbreakable.

The heuristic: How stable is this partiality? Policy changes every sprint? Runtime. Type axis that’s fundamental to your domain? Parameters. Security boundary that would end your company if violated? Separate traits. The more permanent and critical the constraint, the earlier it should fail.

Most traits pretend partial capability doesn’t exist, then document it away: “Note: returns error if called on weekends.” That’s not documentation, it’s confession. Honest traits admit their limitations in their signatures. The caller deserves to know what might fail before they call it, not after.

**Micro-traits tell truths; facade traits provide comfort.**  
The truths should be small, sharp, and undeniable: one clear capability, one clear set of laws. Facade traits bundle those truths for ergonomics, for trait objects, for humans. The danger is forgetting which is which. The moment the bundle becomes “the truth” and the micro-traits rot, the lies begin.

**Traits compose algebraically or not at all.**  
Real traits want to be combined. They have identities, they compose cleanly, `A + B` means “A and B” without hidden surprises. Fake traits are just bags of methods that happen to share a keyword. When you try to combine them, you get contradictions, weird edge cases, and “don’t do that” sections in the docs.

**Default methods are theorems, not conveniences.**  
A default implementation is only honest if it’s *mathematically forced* by the required methods and the laws — correct for every lawful implementation, not just the ones you happen to have. Anything else is a time bomb: a lie that compiles today and fails in some future implementation you didn’t imagine.

**Evolution pressure reveals design lies.**  
Every PR that wants “just one more method,” “just one more flag,” “just make this async,” is telling you something about reality your trait refused to admit. You can treat that as annoyance, or as data. Pain is data. Suffering is signal. Evolution pressure is your domain screaming that the trait’s story is incomplete.

**The controversial trait is probably the right trait.**  
The trait that makes people uncomfortable, that exposes how messy the domain really is, that refuses to pretend everything is synchronous, infallible, or context-free — that’s the one most likely to survive contact with production. Traits that make everyone feel safe at first are usually the ones that spread the deepest lies.

**The visceral test: can you feel the trait wanting to exist?**  
Some traits feel inevitable once you see them: of *course* we needed this separation; of *course* this capability stands alone. Others feel like you're forcing the language and the domain to tolerate something unnatural. If you have to keep arguing for a trait's existence, it's probably not a joint in reality — it's a convenience for today.

**The uncomfortable admission: most traits shouldn’t exist.**  
Most traits are premature abstractions, wishful polymorphism, or a refusal to admit “we just have this one concrete thing.” The brave move is to wait. Let repetition hurt. Let partial capability show up as pain. When a trait finally reveals itself, it will feel less like invention and more like discovery — like the program was already shaped that way, and you just gave the shape a name.
