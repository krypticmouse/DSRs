# The `.dsrs` program format

One file = one program. Write declarations first, then exactly one `main`. `//` comments. Whitespace is insignificant except inside `js```…```` code fences. Strings are JSON strings. Reserved words cannot be used as names: `dsrs program caps model sig class enum tool lineage main in out predict cot agent hole seq fork join route retry refine loop else js demos string int float bool map true false null while carry`.

## File skeleton (declarations in any order; `main` last)

````
dsrs 1
program <name>

caps { net:search fs:read }                     // capability ceiling; omit if none

model <name> = "<provider:model>" { temperature 0.2 max_tokens 1024 }
// opts (all optional): base_url "…" temperature N max_tokens N
//   max_tool_iterations N max_retries N retry_base_delay_ms N cache true|false

class <Name> {                                  // struct type, referenced by name
  "optional class docs"
  <field>: <type> "optional docs" check("<expr>", "<label>")
}

enum <Name> { <Value> "optional docs" <Value> }  // unit enum

sig <Name> {                                     // an LM-call interface
  "optional instruction — the prompt's task description"
  in  <field>: <type> "optional docs" alias "<lm_name>"
  out <field>: <type> check("<expr>", "<label>") assert("<expr>")
}

tool <name> "<description>" caps [<cap> …] {     // caps [] omitted if none
  in  <field>: <type>
  out <field>: <type>
}                                                // host tool: bound by the runtime
tool <name> "<desc>" { … } js```                 // sandboxed tool: JS in the artifact
(args) => ({ … })
```

lineage { optimizer "…" trainset "…" budget "…" parent "…" date "…" }   // optional

main: <MainSig> = seq { … }                      // the program body; always a seq
````

**Types**: `string` `int` `float` `bool` · `Name` (class/enum) · `"lit"` (literal) · `T[]` (list) · `T?` (optional) · `map<T>` (string-keyed map) · `A | B` (union) · `(A | B)[]` (group). 

## Nodes (inside `seq { … }`)

Every step is `name = <expr>`; names are program-unique. A node may only reference nodes named **earlier**. The seq exports fields with a final `out { … }` step, and `main`'s seq must export every `out` field of `<MainSig>`.

````
name = predict <Sig> @<model> (in1 = <port>, in2 = <port>) { instruction "…" demos [<rows>] render "bare" }
name = cot <Sig> @<model> (…)                    // predict + prepended reasoning output
name = agent <Sig> @<model> (…) {                // LM + tool loop; block required
  tools [<tool> …]  stop_tools [<tool> …]
  max_turns 6  until_parse false
  budget { calls 5 tokens 40000 deadline_ms 60000 on_exhausted finalize }
  context { max_history_turns 4 tool_result_max_bytes 2048 playbook "…" }
  instruction "…"  demos [{"input":{…},"output":{…}}]
}
name = hole <Sig> (…) caps [<cap> …] js```       // typed sandboxed JS; caps [] if none
(a) => ({ out_field: … })
```
name = seq { … out { f = <port> } }              // nested scope
name = fork {                                    // concurrent branches (can't see each other)
  a = <expr>
  b = <expr>
} join { f = a.x, g = b.y }
name = route <port> {                            // port must be enum-typed (or literal union)
  Variant -> leaf_name = predict <Sig> (…)       // arms must export identical fields
  else -> other_name = <expr>                    // else required unless all variants covered
}
name = retry (attempts 3 backoff_ms 100 feedback true) child_name = <expr>
name = refine (threshold 0.8 max_rounds 3 feedback_field <input>) {
  body = child_name = <expr>                     // feedback_field: string input of body
  judge = judge_name = predict <JudgeSig> (…)    // judge outputs: score: float, feedback: string
}
name = loop (max_iters 3) {                      // loops are always bounded
  step_name = predict <Sig> (x = ^field)         // ^field = previous iteration's carry
  while step_name.keep_going                     // optional; bool port
  carry { field = step_name.next }               // field must shadow a scope input
  join { result = step_name.value }              // the loop's exported fields
}
````

`@<model>` may be omitted when exactly one model is declared. Leaf nodes (`predict`/`cot`/`agent`/`hole`) always need a `name =`; containers in arm/child positions may be anonymous.

`render` (predict/cot only) selects the prompt protocol: `"markers"` (default, the `[[ ## field ## ]]` contract — never printed) or `"bare"` (instruction as system prompt, raw input as user turn, whole completion = the output; only valid when the signature has exactly one non-optional `string` output). It is an optimizable slot (`<leaf>.render`) like `instruction` and `demos`.

## Ports (the right side of every binding)

- `$.field` — the enclosing scope's input (program input at top level)
- `node.field` — an output of an earlier-named node
- `^field` — previous loop iteration's carried value (loop bodies only)
- JSON literal — `"text"`, `42`, `1.5`, `true`, `null`, `[…]`, `{…}`

Every `in` field of a leaf's signature must be bound exactly once. Types must match (widenings allowed: `int`→`float`, `T`→`T?`, `T`→union containing `T`).

## Hard rules (violations are compile errors)

1. `dsrs 1` first; `main: <Sig> = seq { … }` last.
2. Node names are program-unique; only earlier nodes are referenceable.
3. Every hole/tool `caps [ … ]` must be a subset of the program `caps { … }`.
4. `route` needs `else` unless its arms cover every enum variant; arms export identical fields.
5. All loops carry explicit bounds (`max_iters`, `max_turns`, `attempts`, `max_rounds`).
6. Signatures need at least one `in` and one `out` field; `check` needs a label.
7. Class/enum/sig/tool/model names must be declared before `main` uses them.

## Minimal complete example

````
dsrs 1
program qa

caps { net:search }

model fast = "openai:gpt-4o-mini"
model deep = "anthropic:claude-sonnet-4-5"

sig Main {
  in  question: string
  out answer: string
  out sources: string[]
}

sig Draft {
  "Draft a thorough, factual answer."
  in  question: string
  out answer: string
}

sig Research {
  "Verify the draft against sources; collect URLs."
  in  question: string
  in  draft: string
  out evidence: string[]
}

sig CiteCheck {
  in  draft: string
  in  evidence: string[]
  out answer: string
  out sources: string[]
}

tool search "Web search; returns result snippets with URLs" caps [net:search] {
  in  query: string
  out results: string[]
}

main: Main = seq {
  drafter = cot Draft @deep (question = $.question)
  researcher = agent Research @fast (question = $.question, draft = drafter.answer) {
    tools [search]
    max_turns 6
    budget { tokens 40000 on_exhausted finalize }
  }
  checker = hole CiteCheck (draft = drafter.answer, evidence = researcher.evidence) caps [] js```
(a) => ({
  answer: a.draft,
  sources: a.evidence.filter(e => e.startsWith("http")),
})
```
  out { answer = checker.answer, sources = checker.sources }
}
````
