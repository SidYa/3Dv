// Exports the COM entry points PRIVATE (not in the import library), as COM
// servers should, through the module definition file.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        let def = std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("3DvThumb.def");
        println!("cargo:rustc-cdylib-link-arg=/DEF:{}", def.display());
        println!("cargo:rerun-if-changed=3DvThumb.def");
    }
}
