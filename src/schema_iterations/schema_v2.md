# FFLO — Entities & Relations (v6)

## Entity Types

### ACT STAGE

| Entity Type | Subtype | Domain Note |
|---|---|---|
| `fflo:FoodBusinessOperator` | — | subclass of `prov:Agent`. Actor responsible for or performing the adulteration act |
| `fflo:AdulterationAct` | — | subclass of `prov:Activity`. A specific instance of adulteration occurring over a period of time |
| `fflo:Adulterant` | — | Substance added to or substituted within a food product fraudulently or negligently. NOT a permitted additive — see `fflo:FoodAdditive` |
| `fflo:Adulterant` | `fflo:SyntheticAdulterant` | Lab-produced or industrial chemical used as adulterant (e.g. Sudan dye, melamine, non-permitted colour) |
| `fflo:Adulterant` | `fflo:NaturalAdulterant` | Cheaper natural substance substituted for a higher-value food (e.g. water in milk, brick powder in chilli) |
| `fflo:Adulterant` | `fflo:ContaminantAdulterant` | Substance not intentionally added but present above FSSAI permissible limits (e.g. pesticide residue, heavy metal) |
| `fflo:AdulterationMethod` | — | How adulteration is physically carried out; subclass of `prov:Activity` |
| `fflo:FraudType` | — | Controlled vocabulary: `Dilution` / `Substitution` / `Mislabeling` / `Masking` / `Contamination` |
| `fflo:IntentionalityLevel` | — | Datatype property values on `fflo:AdulterationMethod`: `Intentional` / `Negligent` / `Incidental` |
| `fflo:FoodAdditive` | — | A substance permitted by FSSAI for use in food at or below a specified level, carrying a declared technological function. Subclass of `fkg:ChemicalIngredient`. Becomes an `fflo:Adulterant` only if detected above its `fflo:PermissibleLimit` |
| `fflo:AdditiveFunction` | — | Controlled vocabulary for FSSAI Appendix A functional classes: `fflo:Preservative`, `fflo:Antioxidant`, `fflo:Emulsifier`, `fflo:Stabilizer`, `fflo:AcidityRegulator`, `fflo:FlourTreatmentAgent`, `fflo:Sequestrant`, `fflo:HumectantAdditive`, `fflo:Colourant`, `fflo:SweeteningAgent` |
| `fkg:Food` | — | Primary food entity (Milk, Mustard Oil, Jaggery, Chilli Powder, Atta…) |
| `fkg:Ingredient` | — | Component ingredient of a food |
| `fkg:ChemicalIngredient` | — | Superclass for `fflo:Adulterant` and `fflo:FoodAdditive` |

### SPREAD STAGE

| Entity Type | Notes |
|---|---|
| `fflo:SupplyChainStep` | Superclass for all supply chain nodes |
| `fflo:Production` | Farm, harvest, or primary production level |
| `fflo:Processing` | Manufacturing, milling, packaging, or processing stage |
| `fflo:Storage` | Warehousing, cold chain, or holding stage |
| `fflo:Distribution` | Wholesale, logistics, or inter-market transport stage |
| `fflo:Retail` | Point of sale — shop, mandi, supermarket, quick-commerce |
| `fflo:Import` | Entry point for imported food products |
| `fflo:SpreadEvent` | A supply chain movement event that carries contamination from one node to the next; subclass of `prov:Activity` |
| `fflo:TransformationEvent` | A processing step that transforms a contaminated input into one or more outputs |

### DETECTION STAGE

| Entity Type | Notes |
|---|---|
| `fso:Sample` | Physical food sample collected for laboratory analysis |
| `fso:SampleType` | Classification of sample (raw, processed, retail, import…) |
| `fso:Analysis` | Analytical procedure performed on a sample in a laboratory |
| `fso:AnalysisResult` | Outcome of a laboratory analysis |
| `ssn:Property` | Measurable property being tested (e.g. Ethion concentration, moisture content) |
| `fso:Measurement` | Numeric measurement node. Attach literal value via `fflo:hasNumericValue` |
| `fso:UnitOfMeasure` | Unit of measurement (mg/kg, ppm, % w/w…) |
| `fso:Laboratory` | Facility where formal analysis is performed |
| `fso:Location` | Geographic coordinates of a sample collection point or lab |
| `fflo:IncidentFinding` | Core detection entity. Asserts that adulterant X was found in food Y |
| `fflo:IncidentFinding` | Subtype `fflo:LabConfirmed` — backed by a formal `fso:AnalysisResult` |
| `fflo:IncidentFinding` | Subtype `fflo:FieldDetected` — produced by a rapid or on-site test |
| `fflo:IncidentFinding` | Subtype `fflo:SurveyAggregated` — population-level finding, no individual sample |
| `fflo:IncidentFinding` | Subtype `fflo:RecallTriggered` — regulatory recall decision as detection trigger |
| `fflo:EvidenceSource` | subclass of `prov:Entity`; document, report, or instrument output supporting an IncidentFinding |
| `fflo:DetectionMethod` | Superclass for all detection procedures |
| `fflo:LaboratoryTest` | Requires lab equipment; maps to `fso:Analysis` via `fflo:isPerformedAs` |
| `fflo:FieldTest` | Rapid / on-site test (DART spot test, iodine test, cold water test) |
| `fflo:SensoryTest` | Visual, olfactory, or taste-based indicator |
| `fflo:DetectionIndicator` | Observable signal produced by a detection method (colour change, precipitate, turbidity) |
| `fflo:DetectionKit` | Physical kit required for a field or sensory test |

### REGULATION STAGE

| Entity Type | Notes |
|---|---|
| `fflo:FoodStandard` | A standard bearing normative content uttered by a legislative body |
| `fflo:RegulatoryDocument` | A regulation bearing one or more norms |
| `fflo:RegulatoryBody` | Body with authority to create norms; FSSAI, BIS, AGMARK, state food authorities |
| `fflo:RegulatoryAction` | A public act using power assigned by law |
| `fflo:PermissibleLimit` | Numeric threshold for a substance in a food (MRL, maximum permitted level) |
| `fflo:Violation` | Instance of a permissible limit being exceeded by a specific IncidentFinding |
| `fflo:FoodCategory` | FSSAI commodity classification (Dairy, Oils, Spices, Cereals…). Carries `fflo:hasDefinition`. Supports `fflo:hasSubcategory` hierarchy |
| `fflo:TableReference` | Subclass of `fflo:RegulatoryDocument`. In-document reference (e.g. "Table-2A", "Appendix B", "Schedule I"). Carries `fflo:tableLabel` |

### HEALTH CONSEQUENCE STAGE

| Entity Type | Notes |
|---|---|
| `fflo:HealthEffect` | Pathological bodily process. Use `fflo:CausalHealthEffect` when causal link is established |
| `fflo:CausalHealthEffect` | Subclass of `fflo:HealthEffect`; causal link to adulterant established by evidence |
| `fflo:AcuteEffect` | subclass of `fflo:HealthEffect`; immediate or short-term effect (poisoning, allergic reaction) |
| `fflo:ChronicEffect` | subclass of `fflo:HealthEffect`; long-term or cumulative effect (carcinogenicity, organ damage) |
| `fflo:VulnerablePopulation` | Population group for whom the health effect is especially severe (children, pregnant women, elderly) |

---

## Relations

### ACT STAGE Relations

| Relation | Domain | Range | Notes |
|---|---|---|---|
| `prov:wasAssociatedWith` | `fflo:AdulterationAct` | `fflo:FoodBusinessOperator` | Links the fraud act to its responsible agent |
| `prov:used` | `fflo:AdulterationAct` | `fkg:Food` | The food targeted by the adulteration act |
| `prov:wasGeneratedBy` | `fflo:Adulterant` (in food) | `fflo:AdulterationAct` | The adulterated food product was generated by the act |
| `prov:wasAttributedTo` | `fflo:IncidentFinding` | `fflo:FoodBusinessOperator` | Attributes a finding to a responsible party |
| `fflo:hasAdulterant` | `fkg:Food` | `fflo:Adulterant` | Food is commonly adulterated with this substance |
| `fflo:hasAdulterationMethod` | `fkg:Food` | `fflo:AdulterationMethod` | How adulteration of this food typically occurs |
| `fflo:hasFraudType` | `fflo:AdulterationMethod` | `fflo:FraudType` | Classifies the method into fraud taxonomy |
| `fflo:hasIntentionality` | `fflo:AdulterationMethod` | `fflo:IntentionalityLevel` | Intentional / Negligent / Incidental |
| `fflo:isSubstituteFor` | `fflo:Adulterant` | `fkg:Food` | FKG-inspired; FFLO namespace |
| `fflo:commonIn` | `fflo:Adulterant` | `fflo:GeographicRegion` | Geographic prevalence of this adulterant |
| `fflo:foundAt` | `fflo:Adulterant` | `fflo:SupplyChainStep` | Supply chain node where adulteration typically occurs |
| `fflo:hasFunction` | `fflo:FoodAdditive` | `fflo:AdditiveFunction` | Declares the technological function of a permitted additive (e.g. `Sulphur dioxide → fflo:Preservative`). Do NOT use `hasAdditiveFunction`, `hasTechnologicalFunction`, `hasTechnicalFunction` — always use `fflo:hasFunction` |
| `fkg:hasIngredient` | `fkg:Food` | `fkg:Ingredient` | Declares that a food contains this ingredient. Always use `fkg:hasIngredient`, never `fflo:hasIngredient` |

### SPREAD STAGE Relations

| Relation | Domain | Range | Notes |
|---|---|---|---|
| `prov:wasInfluencedBy` | `fflo:SpreadEvent` | `fflo:AdulterationAct` | Spread event was triggered by an upstream act |
| `prov:wasGeneratedBy` | `fflo:SpreadEvent` | `fflo:TransformationEvent` | Spread product generated by a transformation |
| `fflo:propagatesTo` | `fflo:SupplyChainStep` | `fflo:SupplyChainStep` | Heuristic spread edge: contamination moves from one node to the next |
| `fflo:carriedBy` | `fflo:SpreadEvent` | `fkg:Food` | The food product carrying the contamination |
| `fflo:occursAt` | `fflo:SpreadEvent` | `fflo:SupplyChainStep` | The supply chain node at which the spread event occurs |
| `fflo:inputTo` | `fkg:Food` | `fflo:TransformationEvent` | Contaminated food entering a processing step |
| `fflo:outputOf` | `fkg:Food` | `fflo:TransformationEvent` | Adulterated product produced by a processing step |

### DETECTION STAGE Relations

| Relation | Domain | Range | Notes |
|---|---|---|---|
| `fso:isPerformedOn` | `fso:Analysis` | `fso:Sample` | — |
| `fso:isPerformedAt` | `fso:Analysis` | `fso:Laboratory` | — |
| `fso:isResultOf` | `fso:AnalysisResult` | `fso:Analysis` | — |
| `fso:relatesToProperty` | `fso:AnalysisResult` | `ssn:Property` | — |
| `fso:isMeasuredIn` | `fso:Measurement` | `fso:UnitOfMeasure` | — |
| `fso:hasSampleType` | `fso:Sample` | `fso:SampleType` | — |
| `fso:hasLocation` | `fso:Sample` | `fso:Location` | — |
| `fflo:hasNumericValue` | `fso:Measurement` | `xsd:decimal` | Attaches a literal numeric value to a Measurement node. Do NOT use `fso:hasNumericValue` — it does not exist |
| `fflo:identifies` | `fflo:IncidentFinding` | `fflo:Adulterant` | What adulterant was found |
| `fflo:foundIn` | `fflo:IncidentFinding` | `fkg:Food` | Which food product it was found in |
| `fflo:producedBy` | `fflo:IncidentFinding` | `fflo:DetectionMethod` | What method produced the finding |
| `fflo:confirmedBy` | `fflo:IncidentFinding` | `fso:AnalysisResult` | Optional — LabConfirmed subtype only |
| `fflo:supportedBy` | `fflo:IncidentFinding` | `fflo:EvidenceSource` | Document or report backing the finding |
| `fflo:foundAtStep` | `fflo:IncidentFinding` | `fflo:SupplyChainStep` | Where in the supply chain it was detected |
| `fflo:inRegion` | `fflo:IncidentFinding` | `fflo:GeographicRegion` | Geographic scope of the finding |
| `fflo:triggeredAction` | `fflo:IncidentFinding` | `fflo:RegulatoryAction` | Links detection to its regulatory outcome |
| `fflo:collectedAt` | `fso:Sample` | `fflo:SupplyChainStep` | Bridges FSO sample to supply chain node |
| `fflo:detectedBy` | `fflo:Adulterant` | `fflo:DetectionMethod` | Which methods can detect this adulterant |
| `fflo:producesIndicator` | `fflo:DetectionMethod` | `fflo:DetectionIndicator` | Observable signal the method produces |
| `fflo:requiresKit` | `fflo:DetectionMethod` | `fflo:DetectionKit` | Physical kit required |
| `fflo:isPerformedAs` | `fflo:LaboratoryTest` | `fso:Analysis` | Aligns LaboratoryTest subtype to FSO Analysis chain |

### REGULATION STAGE Relations

| Relation | Domain | Range | Notes |
|---|---|---|---|
| `lkif:created_by` | `fflo:FoodStandard` | `fflo:RegulatoryBody` | — |
| `fflo:belongsToCategory` | `fkg:Food` | `fflo:FoodCategory` | Bridges FKG food entity to FSSAI classification |
| `fflo:appliesToCategory` | `fflo:FoodStandard` | `fflo:FoodCategory` | Links a standard to its FSSAI food category |
| `fflo:hasValue` | `fflo:PermissibleLimit` | `fso:Measurement` | Grounds regulatory limit to FSO Measurement |
| `fflo:forProperty` | `fflo:PermissibleLimit` | `ssn:Property` | Grounds limit to the measured property |
| `fflo:comparedAgainst` | `fso:AnalysisResult` | `fflo:PermissibleLimit` | Violation confirmation — FSO result vs FSSAI threshold |
| `fflo:definedIn` | `fflo:FoodStandard` | `fflo:RegulatoryDocument` | Standard appears in this document |
| `fflo:defines` | `fflo:RegulatoryDocument` | `fflo:FoodCategory` ∪ `fflo:FoodStandard` | Inverse of `fflo:definedIn`. Use when the document is the subject (e.g. "Appendix B.1 defines Fermented meat product") |
| `fflo:appliesTo` | `fflo:PermissibleLimit` | `fflo:Adulterant` | Limit is for this substance. Direction matters: PermissibleLimit is always the subject, never the object |
| `fflo:inFood` | `fflo:PermissibleLimit` | `fkg:Food` | Limit applies in this food |
| `fflo:hasPermissibleLimit` | `fkg:Food` ∪ `fkg:Ingredient` | `fflo:PermissibleLimit` | Inverse of `fflo:inFood`. Use when the food is the subject (e.g. "Whey has permissible limit of 100 ml/litre") |
| `fflo:specifiedIn` | `fflo:PermissibleLimit` | `fflo:FoodStandard` ∪ `fflo:TableReference` | Limit is defined in this standard or table reference |
| `fflo:appliesToFood` | `fflo:FoodStandard` ∪ `fflo:RegulatoryDocument` | `fkg:Food` | Direct standard → food edge. Use only when no food category node is named in the text |
| `fflo:compliesWith` | `fflo:FoodBusinessOperator` ∪ `fkg:Food` | `fflo:FoodStandard` ∪ `fflo:RegulatoryDocument` | Regulatory compliance obligation. Do NOT use `fflo:conformsTo` |
| `fflo:amends` | `fflo:RegulatoryDocument` | `fflo:RegulatoryDocument` | Models the amendment chain between FSSAI regulation versions |
| `fflo:hasSubcategory` | `fflo:FoodCategory` | `fflo:FoodCategory` | Transitive. FSSAI numeric category hierarchy (e.g. 10.0 → 10.4). Do NOT use `fflo:hasSubCategory` |
| `fflo:hasDefinition` | `fflo:FoodCategory` ∪ `fflo:Adulterant` | `xsd:string` | Verbatim FSSAI definitional text. Do NOT use `fflo:definition`, `fflo:definedAs`, `fflo:hasDescription` |
| `fflo:tableLabel` | `fflo:TableReference` | `xsd:string` | Label of a table/appendix reference (e.g. "Table-2A") |
| `fflo:issuedBy` | `fflo:RegulatoryAction` | `fflo:RegulatoryBody` | Which body issued the action |
| `fflo:targets` | `fflo:RegulatoryAction` | `fkg:Food` | Food product subject to the action |
| `fflo:citesFinding` | `fflo:RegulatoryAction` | `fflo:IncidentFinding` | The finding that triggered the regulatory action |
| `fflo:constitutes` | `fflo:IncidentFinding` | `fflo:Violation` | Finding constitutes a violation of a standard |
| `fflo:violates` | `fflo:Violation` | `fflo:PermissibleLimit` | Which permissible limit was violated |

### HEALTH CONSEQUENCE STAGE Relations

| Relation | Domain | Range | Notes |
|---|---|---|---|
| `fflo:causesEffect` | `fflo:Adulterant` | `fflo:HealthEffect` | Adulterant causes this health effect |
| `fflo:affectsPopulation` | `fflo:HealthEffect` | `fflo:VulnerablePopulation` | Effect is especially severe for this population |
| `fflo:hasEffectType` | `fflo:HealthEffect` | `fflo:AcuteEffect` / `fflo:ChronicEffect` | Classifies the effect by temporality |
| `fflo:associatedWith` | `fflo:HealthEffect` | `fflo:IncidentFinding` | Links a health outcome to a specific incident |
