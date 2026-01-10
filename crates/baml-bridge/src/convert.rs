use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    rc::Rc,
    sync::Arc,
};

use baml_types::{BamlMap, BamlValue};

#[derive(Debug, Clone)]
pub struct BamlConvertError {
    pub path: Vec<String>,
    pub expected: &'static str,
    pub got: String,
    pub message: String,
}

impl BamlConvertError {
    pub fn new(
        path: Vec<String>,
        expected: &'static str,
        got: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            path,
            expected,
            got: got.into(),
            message: message.into(),
        }
    }

    pub fn with_path(mut self, segment: impl Into<String>) -> Self {
        self.path.push(segment.into());
        self
    }

    pub fn path_string(&self) -> String {
        if self.path.is_empty() {
            "<root>".to_string()
        } else {
            self.path.join(".")
        }
    }
}

impl fmt::Display for BamlConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (expected {}, got {}) at {}",
            self.message,
            self.expected,
            self.got,
            self.path_string()
        )
    }
}

impl std::error::Error for BamlConvertError {}

pub trait BamlValueConvert: Sized {
    fn try_from_baml_value(value: BamlValue, path: Vec<String>) -> Result<Self, BamlConvertError>;
}

impl BamlValueConvert for String {
    fn try_from_baml_value(value: BamlValue, path: Vec<String>) -> Result<Self, BamlConvertError> {
        match value {
            BamlValue::String(s) => Ok(s),
            other => Err(BamlConvertError::new(
                path,
                "string",
                format!("{other:?}"),
                "expected a string",
            )),
        }
    }
}

impl BamlValueConvert for bool {
    fn try_from_baml_value(value: BamlValue, path: Vec<String>) -> Result<Self, BamlConvertError> {
        match value {
            BamlValue::Bool(v) => Ok(v),
            other => Err(BamlConvertError::new(
                path,
                "bool",
                format!("{other:?}"),
                "expected a boolean",
            )),
        }
    }
}

impl BamlValueConvert for f64 {
    fn try_from_baml_value(value: BamlValue, path: Vec<String>) -> Result<Self, BamlConvertError> {
        match value {
            BamlValue::Float(v) => Ok(v),
            BamlValue::Int(v) => Ok(v as f64),
            other => Err(BamlConvertError::new(
                path,
                "float",
                format!("{other:?}"),
                "expected a float",
            )),
        }
    }
}

impl BamlValueConvert for f32 {
    fn try_from_baml_value(value: BamlValue, path: Vec<String>) -> Result<Self, BamlConvertError> {
        let path_clone = path.clone();
        let v = f64::try_from_baml_value(value, path)?;
        if v > f32::MAX as f64 || v < f32::MIN as f64 {
            return Err(BamlConvertError::new(
                path_clone,
                "float",
                v.to_string(),
                "float out of range",
            ));
        }
        Ok(v as f32)
    }
}

impl BamlValueConvert for i64 {
    fn try_from_baml_value(value: BamlValue, path: Vec<String>) -> Result<Self, BamlConvertError> {
        match value {
            BamlValue::Int(v) => Ok(v),
            other => Err(BamlConvertError::new(
                path,
                "int",
                format!("{other:?}"),
                "expected an integer",
            )),
        }
    }
}

macro_rules! impl_signed_int {
    ($ty:ty) => {
        impl BamlValueConvert for $ty {
            fn try_from_baml_value(
                value: BamlValue,
                path: Vec<String>,
            ) -> Result<Self, BamlConvertError> {
                let v = i64::try_from_baml_value(value, path.clone())?;
                if v < <$ty>::MIN as i64 || v > <$ty>::MAX as i64 {
                    return Err(BamlConvertError::new(
                        path,
                        stringify!($ty),
                        v.to_string(),
                        "integer out of range",
                    ));
                }
                Ok(v as $ty)
            }
        }
    };
}

impl_signed_int!(i8);
impl_signed_int!(i16);
impl_signed_int!(i32);
impl_signed_int!(isize);

macro_rules! impl_unsigned_int {
    ($ty:ty) => {
        impl BamlValueConvert for $ty {
            fn try_from_baml_value(
                value: BamlValue,
                path: Vec<String>,
            ) -> Result<Self, BamlConvertError> {
                let v = i64::try_from_baml_value(value, path.clone())?;
                if v < 0 || v > <$ty>::MAX as i64 {
                    return Err(BamlConvertError::new(
                        path,
                        stringify!($ty),
                        v.to_string(),
                        "integer out of range",
                    ));
                }
                Ok(v as $ty)
            }
        }
    };
}

impl_unsigned_int!(u8);
impl_unsigned_int!(u16);
impl_unsigned_int!(u32);

impl<T> BamlValueConvert for Option<T>
where
    T: BamlValueConvert,
{
    fn try_from_baml_value(value: BamlValue, path: Vec<String>) -> Result<Self, BamlConvertError> {
        match value {
            BamlValue::Null => Ok(None),
            other => Ok(Some(T::try_from_baml_value(other, path)?)),
        }
    }
}

impl<T> BamlValueConvert for Vec<T>
where
    T: BamlValueConvert,
{
    fn try_from_baml_value(value: BamlValue, path: Vec<String>) -> Result<Self, BamlConvertError> {
        match value {
            BamlValue::List(items) => items
                .into_iter()
                .enumerate()
                .map(|(idx, item)| {
                    let mut item_path = path.clone();
                    item_path.push(idx.to_string());
                    T::try_from_baml_value(item, item_path)
                })
                .collect(),
            other => Err(BamlConvertError::new(
                path,
                "list",
                format!("{other:?}"),
                "expected a list",
            )),
        }
    }
}

impl<T> BamlValueConvert for Box<T>
where
    T: BamlValueConvert,
{
    fn try_from_baml_value(value: BamlValue, path: Vec<String>) -> Result<Self, BamlConvertError> {
        Ok(Box::new(T::try_from_baml_value(value, path)?))
    }
}

impl<T> BamlValueConvert for Arc<T>
where
    T: BamlValueConvert,
{
    fn try_from_baml_value(value: BamlValue, path: Vec<String>) -> Result<Self, BamlConvertError> {
        Ok(Arc::new(T::try_from_baml_value(value, path)?))
    }
}

impl<T> BamlValueConvert for Rc<T>
where
    T: BamlValueConvert,
{
    fn try_from_baml_value(value: BamlValue, path: Vec<String>) -> Result<Self, BamlConvertError> {
        Ok(Rc::new(T::try_from_baml_value(value, path)?))
    }
}

impl<V> BamlValueConvert for HashMap<String, V>
where
    V: BamlValueConvert,
{
    fn try_from_baml_value(value: BamlValue, path: Vec<String>) -> Result<Self, BamlConvertError> {
        map_to_collection(value, path, |map| map.into_iter().collect())
    }
}

impl<V> BamlValueConvert for BTreeMap<String, V>
where
    V: BamlValueConvert,
{
    fn try_from_baml_value(value: BamlValue, path: Vec<String>) -> Result<Self, BamlConvertError> {
        map_to_collection(value, path, |map| map.into_iter().collect())
    }
}

fn map_to_collection<V, C>(
    value: BamlValue,
    path: Vec<String>,
    to_collection: impl FnOnce(HashMap<String, V>) -> C,
) -> Result<C, BamlConvertError>
where
    V: BamlValueConvert,
{
    let map = match value {
        BamlValue::Map(map) => map,
        other => {
            return Err(BamlConvertError::new(
                path,
                "map",
                format!("{other:?}"),
                "expected a map",
            ))
        }
    };

    let mut out = HashMap::new();
    for (key, value) in map.into_iter() {
        let mut item_path = path.clone();
        item_path.push(key.clone());
        let converted = V::try_from_baml_value(value, item_path)?;
        out.insert(key, converted);
    }

    Ok(to_collection(out))
}

pub fn get_field<'a>(
    map: &'a BamlMap<String, BamlValue>,
    name: &str,
    alias: Option<&str>,
) -> Option<&'a BamlValue> {
    map.get(name)
        .or_else(|| alias.and_then(|alias| map.get(alias)))
}
