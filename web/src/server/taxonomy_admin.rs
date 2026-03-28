//! Taxonomy admin helpers for the web UI.
//!
//! Provides safe read/update operations for the taxonomy equivalence TOML.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use crate::model::TaxonomyAdminStatus;

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct TaxonomyFile {
    #[serde(default)]
    species: Vec<SpeciesAliases>,
    #[serde(default)]
    class_aliases: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct SpeciesAliases {
    canonical: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    class: Option<String>,
}

fn normalize_sci_name(raw: &str) -> String {
    let replaced = raw.replace('_', " ");
    let words: Vec<&str> = replaced.split_whitespace().collect();
    if words.is_empty() {
        return String::new();
    }
    let mut result = String::with_capacity(raw.len());
    for (i, word) in words.iter().enumerate() {
        if i > 0 {
            result.push(' ');
        }
        if i == 0 {
            let mut chars = word.chars();
            if let Some(first) = chars.next() {
                result.extend(first.to_uppercase());
                result.extend(chars.map(|c| c.to_ascii_lowercase()));
            }
        } else {
            result.extend(word.chars().map(|c| c.to_ascii_lowercase()));
        }
    }
    result
}

fn normalize_class(raw: &str) -> String {
    raw.trim().to_ascii_lowercase()
}

fn taxonomy_path() -> PathBuf {
    if let Ok(p) = std::env::var("GAIA_TAXONOMY_TABLE") {
        if !p.trim().is_empty() {
            return PathBuf::from(p);
        }
    }

    let models_path = PathBuf::from("/models/_taxonomy/taxonomy_equivalences.toml");
    if models_path.exists() {
        return models_path;
    }

    PathBuf::from("/data/_taxonomy/taxonomy_equivalences.toml")
}

fn default_taxonomy_toml() -> TaxonomyFile {
    let mut class_aliases = BTreeMap::new();
    class_aliases.insert("aves".to_string(), "birds".to_string());
    class_aliases.insert("bird".to_string(), "birds".to_string());
    class_aliases.insert("birds".to_string(), "birds".to_string());
    class_aliases.insert("wildlife".to_string(), "wildlife".to_string());

    TaxonomyFile {
        species: vec![SpeciesAliases {
            canonical: "Cyanocorax morio".to_string(),
            aliases: vec!["Psilorhinus morio".to_string()],
            class: Some("birds".to_string()),
        }],
        class_aliases,
    }
}

fn read_or_init_taxonomy(path: &Path) -> Result<TaxonomyFile, String> {
    if path.exists() {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("Cannot read taxonomy file {}: {e}", path.display()))?;
        toml::from_str::<TaxonomyFile>(&text)
            .map_err(|e| format!("Invalid taxonomy TOML {}: {e}", path.display()))
    } else {
        Ok(default_taxonomy_toml())
    }
}

fn write_taxonomy_atomic(path: &Path, tax: &TaxonomyFile) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create taxonomy directory {}: {e}", parent.display()))?;
    }

    let text = toml::to_string_pretty(tax)
        .map_err(|e| format!("Cannot serialize taxonomy TOML: {e}"))?;

    let tmp = path.with_extension("toml.tmp");
    std::fs::write(&tmp, text)
        .map_err(|e| format!("Cannot write temporary taxonomy file {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .map_err(|e| format!("Cannot replace taxonomy file {}: {e}", path.display()))?;

    Ok(())
}

pub fn taxonomy_status() -> Result<TaxonomyAdminStatus, String> {
    let path = taxonomy_path();
    let exists = path.exists();

    let writable = if exists {
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .is_ok()
    } else {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
            let probe = parent.join(".taxonomy-write-test");
            match std::fs::write(&probe, b"ok") {
                Ok(_) => {
                    let _ = std::fs::remove_file(probe);
                    true
                }
                Err(_) => false,
            }
        } else {
            false
        }
    };

    let tax = read_or_init_taxonomy(&path)?;
    let species_count = tax.species.len() as u32;
    let alias_count = tax.species.iter().map(|s| s.aliases.len() as u32).sum();
    let class_alias_count = tax.class_aliases.len() as u32;

    Ok(TaxonomyAdminStatus {
        path: path.display().to_string(),
        exists,
        writable,
        species_count,
        alias_count,
        class_alias_count,
    })
}

pub fn upsert_species_alias(canonical: &str, alias: &str, class_name: Option<&str>) -> Result<String, String> {
    let canonical = normalize_sci_name(canonical);
    let alias = normalize_sci_name(alias);

    if canonical.is_empty() {
        return Err("Canonical scientific name is required".to_string());
    }
    if alias.is_empty() {
        return Err("Alias scientific name is required".to_string());
    }
    if canonical == alias {
        return Err("Canonical and alias names must be different".to_string());
    }

    let path = taxonomy_path();
    let mut tax = read_or_init_taxonomy(&path)?;

    // Conflict check: alias cannot point to a different canonical.
    for sp in &tax.species {
        let sp_canonical = normalize_sci_name(&sp.canonical);
        if sp_canonical != canonical {
            if sp_canonical == alias {
                return Err(format!(
                    "Alias '{}' is already a canonical name for '{}'",
                    alias, sp_canonical
                ));
            }
            if sp.aliases.iter().any(|a| normalize_sci_name(a) == alias) {
                return Err(format!(
                    "Alias '{}' is already mapped to canonical '{}'",
                    alias, sp_canonical
                ));
            }
        }
    }

    let mut updated = false;
    for sp in &mut tax.species {
        if normalize_sci_name(&sp.canonical) == canonical {
            if !sp.aliases.iter().any(|a| normalize_sci_name(a) == alias) {
                sp.aliases.push(alias.clone());
                sp.aliases.sort();
                sp.aliases.dedup();
            }
            if let Some(cls) = class_name {
                let cls = normalize_class(cls);
                if !cls.is_empty() {
                    sp.class = Some(cls);
                }
            }
            updated = true;
            break;
        }
    }

    if !updated {
        let cls = class_name
            .map(normalize_class)
            .filter(|c| !c.is_empty());
        tax.species.push(SpeciesAliases {
            canonical,
            aliases: vec![alias],
            class: cls,
        });
        tax.species
            .sort_by(|a, b| normalize_sci_name(&a.canonical).cmp(&normalize_sci_name(&b.canonical)));
    }

    write_taxonomy_atomic(&path, &tax)?;
    Ok(format!("Saved taxonomy alias in {}", path.display()))
}

pub fn upsert_class_alias(alias: &str, canonical_class: &str) -> Result<String, String> {
    let alias = normalize_class(alias);
    let canonical_class = normalize_class(canonical_class);

    if alias.is_empty() {
        return Err("Class alias is required".to_string());
    }
    if canonical_class.is_empty() {
        return Err("Canonical class is required".to_string());
    }

    let path = taxonomy_path();
    let mut tax = read_or_init_taxonomy(&path)?;
    tax.class_aliases.insert(alias, canonical_class);
    write_taxonomy_atomic(&path, &tax)?;

    Ok(format!("Saved class alias in {}", path.display()))
}
