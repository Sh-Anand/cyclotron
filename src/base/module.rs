use crate::base::behavior::*;
use std::sync::{Arc, OnceLock};

#[derive(Debug)]
pub struct ModuleBase<T, C> {
    pub state: T,
    pub config: OnceLock<Arc<C>>,
}

impl<T: Default, C: Default> Default for ModuleBase<T, C> {
    fn default() -> Self {
        Self {
            state: T::default(),
            config: OnceLock::new(),
        }
    }
}

pub trait IsModule: ModuleBehaviors {
    type StateType;
    type ConfigType;

    fn base(&mut self) -> &mut ModuleBase<Self::StateType, Self::ConfigType>;

    fn base_ref(&self) -> &ModuleBase<Self::StateType, Self::ConfigType>;

    fn state_mut(&mut self) -> &mut Self::StateType {
        &mut self.base().state
    }

    fn state(&self) -> &Self::StateType {
        &self.base_ref().state
    }
}

impl<X> Parameterizable for X
where
    X: IsModule,
{
    type ConfigType = X::ConfigType;

    fn conf(&self) -> &Self::ConfigType {
        self.base_ref()
            .config
            .get()
            .expect("config not found, was `init_conf` called in `new`?")
    }

    fn init_conf(&mut self, conf: Arc<Self::ConfigType>) {
        self.base()
            .config
            .set(Arc::clone(&conf))
            .map_err(|_| "config already set")
            .unwrap();
    }
}

/// arguments: identifier, state type, config type, additional methods
macro_rules! module {
    ($comp:ident, $T:ty, $C:ty, $($method:item)*) => {
        impl IsModule for $comp {
            type StateType = $T;
            type ConfigType = $C;

            fn base(&mut self) -> &mut ModuleBase<$T, $C> {
                &mut self.base
            }

            fn base_ref(&self) -> &ModuleBase<$T, $C> {
                &self.base
            }

            $($method)*
        }
    };
}

pub(crate) use module;
