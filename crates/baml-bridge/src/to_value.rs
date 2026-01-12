use std::{
    collections::{BTreeMap, HashMap},
    rc::Rc,
    sync::Arc,
};

use baml_types::{BamlMap, BamlValue};

pub trait ToBamlValue {
    fn to_baml_value(&self) -> BamlValue;
}

impl<T: ToBamlValue + ?Sized> ToBamlValue for &T {
    fn to_baml_value(&self) -> BamlValue {
        (*self).to_baml_value()
    }
}

impl ToBamlValue for BamlValue {
    fn to_baml_value(&self) -> BamlValue {
        self.clone()
    }
}

impl ToBamlValue for String {
    fn to_baml_value(&self) -> BamlValue {
        BamlValue::String(self.clone())
    }
}

impl ToBamlValue for &str {
    fn to_baml_value(&self) -> BamlValue {
        BamlValue::String((*self).to_string())
    }
}

impl ToBamlValue for bool {
    fn to_baml_value(&self) -> BamlValue {
        BamlValue::Bool(*self)
    }
}

impl ToBamlValue for i64 {
    fn to_baml_value(&self) -> BamlValue {
        BamlValue::Int(*self)
    }
}

macro_rules! impl_signed_int {
    ($ty:ty) => {
        impl ToBamlValue for $ty {
            fn to_baml_value(&self) -> BamlValue {
                BamlValue::Int(*self as i64)
            }
        }
    };
}

macro_rules! impl_unsigned_int {
    ($ty:ty) => {
        impl ToBamlValue for $ty {
            fn to_baml_value(&self) -> BamlValue {
                BamlValue::Int(*self as i64)
            }
        }
    };
}

impl_signed_int!(i8);
impl_signed_int!(i16);
impl_signed_int!(i32);
impl_signed_int!(isize);

impl_unsigned_int!(u8);
impl_unsigned_int!(u16);
impl_unsigned_int!(u32);

impl ToBamlValue for f32 {
    fn to_baml_value(&self) -> BamlValue {
        BamlValue::Float(*self as f64)
    }
}

impl ToBamlValue for f64 {
    fn to_baml_value(&self) -> BamlValue {
        BamlValue::Float(*self)
    }
}

impl ToBamlValue for baml_types::BamlMedia {
    fn to_baml_value(&self) -> BamlValue {
        BamlValue::Media(self.clone())
    }
}

impl<T: ToBamlValue> ToBamlValue for Option<T> {
    fn to_baml_value(&self) -> BamlValue {
        match self {
            Some(value) => value.to_baml_value(),
            None => BamlValue::Null,
        }
    }
}

impl<T: ToBamlValue> ToBamlValue for Vec<T> {
    fn to_baml_value(&self) -> BamlValue {
        BamlValue::List(self.iter().map(|value| value.to_baml_value()).collect())
    }
}

impl<T: ToBamlValue> ToBamlValue for HashMap<String, T> {
    fn to_baml_value(&self) -> BamlValue {
        let mut map: BamlMap<String, BamlValue> = BamlMap::new();
        for (key, value) in self {
            map.insert(key.clone(), value.to_baml_value());
        }
        BamlValue::Map(map)
    }
}

impl<T: ToBamlValue> ToBamlValue for BTreeMap<String, T> {
    fn to_baml_value(&self) -> BamlValue {
        let mut map: BamlMap<String, BamlValue> = BamlMap::new();
        for (key, value) in self {
            map.insert(key.clone(), value.to_baml_value());
        }
        BamlValue::Map(map)
    }
}

impl<T: ToBamlValue> ToBamlValue for Box<T> {
    fn to_baml_value(&self) -> BamlValue {
        (**self).to_baml_value()
    }
}

impl<T: ToBamlValue> ToBamlValue for Arc<T> {
    fn to_baml_value(&self) -> BamlValue {
        (**self).to_baml_value()
    }
}

impl<T: ToBamlValue> ToBamlValue for Rc<T> {
    fn to_baml_value(&self) -> BamlValue {
        (**self).to_baml_value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::rc::Rc;
    use std::sync::Arc;

    #[test]
    fn test_primitive_to_baml_value() {
        assert_eq!("hello".to_baml_value(), BamlValue::String("hello".into()));
        assert_eq!(true.to_baml_value(), BamlValue::Bool(true));
        assert_eq!(42i32.to_baml_value(), BamlValue::Int(42));
        assert_eq!((-7i16).to_baml_value(), BamlValue::Int(-7));
        assert_eq!(3.5f32.to_baml_value(), BamlValue::Float(3.5));
        assert_eq!(2.25f64.to_baml_value(), BamlValue::Float(2.25));
    }

    #[test]
    fn test_container_to_baml_value() {
        let some: Option<i32> = Some(1);
        let none: Option<i32> = None;
        assert_eq!(some.to_baml_value(), BamlValue::Int(1));
        assert_eq!(none.to_baml_value(), BamlValue::Null);

        let vec = vec!["a", "b"];
        assert_eq!(
            vec.to_baml_value(),
            BamlValue::List(vec![
                BamlValue::String("a".into()),
                BamlValue::String("b".into())
            ])
        );

        let mut map: HashMap<String, i32> = HashMap::new();
        map.insert("answer".to_string(), 42);
        let mut expected: BamlMap<String, BamlValue> = BamlMap::new();
        expected.insert("answer".to_string(), BamlValue::Int(42));
        assert_eq!(map.to_baml_value(), BamlValue::Map(expected));

        let mut tree: BTreeMap<String, i32> = BTreeMap::new();
        tree.insert("alpha".to_string(), 1);
        let mut expected_tree: BamlMap<String, BamlValue> = BamlMap::new();
        expected_tree.insert("alpha".to_string(), BamlValue::Int(1));
        assert_eq!(tree.to_baml_value(), BamlValue::Map(expected_tree));

        let boxed: Box<i32> = Box::new(7);
        assert_eq!(boxed.to_baml_value(), BamlValue::Int(7));

        let shared = Arc::new("shared".to_string());
        assert_eq!(
            shared.to_baml_value(),
            BamlValue::String("shared".into())
        );

        let rc_value = Rc::new(true);
        assert_eq!(rc_value.to_baml_value(), BamlValue::Bool(true));
    }
}
