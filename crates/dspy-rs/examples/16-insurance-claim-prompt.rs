/*
Prints a typed system prompt for insurance claim extraction.

Run with:
cargo run --example 16-insurance-claim-prompt
*/

use dspy_rs::{BamlType, ChatAdapter, Signature, init_tracing};

// Keep the example self-contained; dates are represented as YYYY-MM-DD strings.
type NaiveDate = String;

/// Basic claim information (metadata about the claim intake).
#[derive(Debug, Clone, PartialEq, Eq)]
#[BamlType]
pub struct ClaimHeader {
    /// Claim ID in format `CLM-XXXXXX`, where `X` is a digit.
    pub claim_id: Option<String>,

    /// Date claim was reported in `YYYY-MM-DD` format.
    pub report_date: Option<NaiveDate>,

    /// Date incident occurred in `YYYY-MM-DD` format.
    pub incident_date: Option<NaiveDate>,

    /// Full name of person reporting claim.
    pub reported_by: Option<String>,

    /// Channel used to report claim.
    pub channel: Option<ClaimChannel>,
}

/// Channel used to report a claim.
#[derive(Debug, Clone, PartialEq, Eq)]
#[BamlType]
pub enum ClaimChannel {
    Email,
    Phone,
    Portal,
    InPerson,
}

/// Policy information if available.
#[derive(Debug, Clone, PartialEq, Eq)]
#[BamlType]
pub struct PolicyDetails {
    /// Policy number in format `POL-XXXXXXXXX`, where `X` is a digit.
    pub policy_number: Option<String>,

    /// Full legal name on policy.
    pub policyholder_name: Option<String>,

    /// Type of insurance coverage.
    pub coverage_type: Option<CoverageType>,

    /// Policy effective start date in `YYYY-MM-DD` format.
    pub effective_date: Option<NaiveDate>,

    /// Policy expiration end date in `YYYY-MM-DD` format.
    pub expiration_date: Option<NaiveDate>,
}

/// Type of insurance coverage.
#[derive(Debug, Clone, PartialEq, Eq)]
#[BamlType]
pub enum CoverageType {
    Property,
    Auto,
    Liability,
    Health,
    Travel,
    Other,
}

/// An insured object involved in the claim (vehicle, building, person, etc.).
#[derive(Debug, Clone, PartialEq, Eq)]
#[BamlType]
pub struct InsuredObject {
    /// Unique identifier for insured object.
    ///
    /// For vehicles, use VIN format (e.g., `VIN12345678901234567`).
    /// For buildings, use `PROP-XXXXXX` format.
    /// For liability, use `LIAB-XXXXXX` format.
    /// For other objects, use `OBJ-XXXXXX` format,
    /// where `X` is a digit.
    pub object_id: Option<String>,

    /// Type of insured object.
    pub object_type: InsuredObjectType,

    /// Make and model for vehicles (use standardized manufacturer names and models),
    /// or building type for property.
    pub make_model: Option<String>,

    /// Year for vehicles or year built for buildings.
    pub year: Option<i32>,

    /// Full street address where object is located or originated from.
    pub location_address: Option<String>,

    /// Estimated monetary value in USD without currency symbol.
    pub estimated_value: Option<i64>,
}

/// Type of insured object.
#[derive(Debug, Clone, PartialEq, Eq)]
#[BamlType]
pub enum InsuredObjectType {
    Vehicle,
    Building,
    Person,
    Other,
}

/// Structured incident details.
#[derive(Debug, Clone, PartialEq, Eq)]
#[BamlType]
pub struct IncidentDescription {
    /// Specific standardized incident type.
    pub incident_type: IncidentType,

    /// Standardized location type where incident occurred.
    pub location_type: LocationType,

    /// Estimated damage in USD without currency symbol.
    pub estimated_damage_amount: Option<i64>,

    /// Police report number if applicable.
    pub police_report_number: Option<String>,
}

/// Specific standardized incident type.
#[derive(Debug, Clone, PartialEq, Eq)]
#[BamlType]
pub enum IncidentType {
    RearEndCollision,
    SideImpactCollision,
    HeadOnCollision,
    ParkingLotCollision,
    HouseFire,
    KitchenFire,
    ElectricalFire,
    BurstPipeFlood,
    StormDamage,
    RoofLeak,
    SlipAndFall,
    PropertyInjury,
    ProductLiability,
    TheftBurglary,
    Vandalism,
}

/// Standardized location type where incident occurred.
#[derive(Debug, Clone, PartialEq, Eq)]
#[BamlType]
pub enum LocationType {
    Intersection,
    Highway,
    ParkingLot,
    Driveway,
    ResidentialStreet,
    ResidenceInterior,
    ResidenceExterior,
    CommercialProperty,
    PublicProperty,
}

/// Top-level insurance claim object aggregating all extracted fields.
#[derive(Debug, Clone, PartialEq, Eq)]
#[BamlType]
pub struct InsuranceClaim {
    /// Basic claim information.
    pub header: ClaimHeader,

    /// Policy information if available.
    pub policy_details: Option<PolicyDetails>,

    /// List of insured objects involved, if applicable.
    pub insured_objects: Option<Vec<InsuredObject>>,

    /// Structured incident details.
    pub incident_description: Option<IncidentDescription>,
}

#[derive(Signature, Clone, Debug)]
/// Extract the insurance claim information from the following text.
/// - If you are unsure about a field, leave it as null.
pub struct InsuranceClaimInfo {
    #[input]
    claim_text: String,

    #[output]
    claim: InsuranceClaim,
}

fn main() {
    init_tracing().expect("failed to initialize tracing");

    let adapter = ChatAdapter;
    let system = adapter
        .format_system_message_typed::<InsuranceClaimInfo>()
        .expect("system prompt");
    let user = adapter.format_user_message_typed::<InsuranceClaimInfo>(&InsuranceClaimInfoInput {
        claim_text: "A raccoon bumped a parked scooter in a driveway. Reported by Taylor P. via phone. No policy details provided.".to_string(),
    });

    println!("=== System ===\n{system}\n");
    println!("=== User ===\n{user}");
}
