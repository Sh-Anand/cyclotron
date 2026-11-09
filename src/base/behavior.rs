use std::sync::Arc;

pub trait ModuleBehaviors {
    fn tick_one(&mut self) {}
    fn reset(&mut self) {}
}

pub trait Parameterizable {
    type ConfigType;

    fn conf(&self) -> &Self::ConfigType;
    fn init_conf(&mut self, conf: Arc<Self::ConfigType>);
}
