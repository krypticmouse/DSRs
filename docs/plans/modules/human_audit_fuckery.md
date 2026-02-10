## purpose
this is a doc that just lists things we gotta look into post phase 4. 

1)
really gotta do some deep research on facet and if we're using it right or not taking advantage of it enough. one of the most likely places things will drift.

2)
legacy and cruft. these are things we want to kill kill kill die die die. as noted in shapes, but i think some of this was unjustly deferred. we gotta fix that shit boss

3)
`#[derive(Facet)]` on module structs is an implementation leak. module authors shouldn't know what Facet is — the concept is "this is a module with discoverable parameters" (like `class MyModule(dspy.Module):` in DSPy). should be `#[derive(Module)]` that implies Facet under the hood. cleanup pass question: new proc macro that emits Facet + possibly validates struct shape, or just a re-export alias? either way the user-facing surface should say Module, not Facet.

4)
