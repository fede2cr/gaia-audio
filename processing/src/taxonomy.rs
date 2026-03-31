//! Scientific-name equivalence and class normalisation.
//!
//! This module lets us reconcile detections from multiple models that
//! use different scientific names for the same species (taxonomy updates,
//! synonyms, legacy labels) and different class/category strings
//! (`AVES`, `BIRDS`, `WILDLIFE`, etc.).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result};
use gaia_common::detection::normalize_sci_name;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default)]
pub struct TaxonomyTable {
    alias_to_canonical: HashMap<String, String>,
    canonical_class: HashMap<String, String>,
    class_aliases: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
struct TaxonomyFile {
    #[serde(default)]
    species: Vec<SpeciesAliases>,
    #[serde(default)]
    class_aliases: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
struct SpeciesAliases {
    canonical: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    class: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct MergedTaxonomyFile {
    species: Vec<MergedSpeciesAliases>,
    class_aliases: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize)]
struct MergedSpeciesAliases {
    canonical: String,
    aliases: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    class: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ReviewReport {
    pub table_path: PathBuf,
    pub merged_path: Option<PathBuf>,
    pub species_count: usize,
    pub alias_count: usize,
    pub class_alias_count: usize,
    pub class_override_count: usize,
}

static TAXONOMY: OnceLock<TaxonomyTable> = OnceLock::new();

impl TaxonomyTable {
    fn empty() -> Self {
        Self::default()
    }

    pub fn load_from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Cannot read taxonomy table: {}", path.display()))?;
        let parsed: TaxonomyFile = toml::from_str(&text)
            .with_context(|| format!("Invalid taxonomy table TOML: {}", path.display()))?;

        let mut table = Self::empty();

        for entry in parsed.species {
            let canonical = normalize_sci_name(&entry.canonical);
            if canonical.is_empty() {
                continue;
            }

            table
                .alias_to_canonical
                .insert(canonical.clone(), canonical.clone());

            if let Some(class_name) = entry.class.as_ref().map(|c| normalize_classification(c)) {
                if !class_name.is_empty() {
                    table.canonical_class.insert(canonical.clone(), class_name);
                }
            }

            for alias in entry.aliases {
                let norm_alias = normalize_sci_name(&alias);
                if norm_alias.is_empty() {
                    continue;
                }
                if let Some(prev) = table.alias_to_canonical.get(&norm_alias) {
                    if prev != &canonical {
                        tracing::warn!(
                            "taxonomy: alias '{}' maps to multiple canonical names ('{}' and '{}'); keeping first",
                            norm_alias,
                            prev,
                            canonical,
                        );
                        continue;
                    }
                }
                table.alias_to_canonical.insert(norm_alias, canonical.clone());
            }
        }

        for (k, v) in parsed.class_aliases {
            let key = k.trim().to_ascii_lowercase();
            let value = v.trim().to_ascii_lowercase();
            if !key.is_empty() && !value.is_empty() {
                table.class_aliases.insert(key, value);
            }
        }

        Ok(table)
    }

    pub fn canonical_species_name(&self, name: &str) -> String {
        let normalized = normalize_sci_name(name);
        self.alias_to_canonical
            .get(&normalized)
            .cloned()
            .unwrap_or(normalized)
    }

    pub fn class_for_species(&self, name: &str) -> Option<String> {
        let canonical = self.canonical_species_name(name);
        self.canonical_class.get(&canonical).cloned()
    }

    pub fn normalize_classification(&self, raw: &str) -> String {
        let key = raw.trim().to_ascii_lowercase();
        if key.is_empty() {
            return String::new();
        }
        self.class_aliases.get(&key).cloned().unwrap_or(key)
    }
}

fn default_table_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("GAIA_TAXONOMY_TABLE") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }

    // Shared data volume path (editable by gaia-web admin UI).
    let data_merged = PathBuf::from("/data/_taxonomy/taxonomy_merged.toml");
    if data_merged.exists() {
        return Some(data_merged);
    }
    let data_base = PathBuf::from("/data/_taxonomy/taxonomy_equivalences.toml");
    if data_base.exists() {
        return Some(data_base);
    }

    let merged = PathBuf::from("/models/_taxonomy/taxonomy_merged.toml");
    if merged.exists() {
        return Some(merged);
    }

    let base = PathBuf::from("/models/_taxonomy/taxonomy_equivalences.toml");
    if base.exists() {
        return Some(base);
    }

    None
}

pub fn global() -> &'static TaxonomyTable {
    TAXONOMY.get_or_init(|| {
        if let Some(path) = default_table_path() {
            match TaxonomyTable::load_from_file(&path) {
                Ok(t) => {
                    tracing::info!("taxonomy: loaded equivalence table from {}", path.display());
                    return t;
                }
                Err(e) => {
                    tracing::warn!(
                        "taxonomy: failed to load {}: {e:#}; falling back to built-in normalization",
                        path.display()
                    );
                }
            }
        }
        TaxonomyTable::empty()
    })
}

pub fn canonical_species_name(name: &str) -> String {
    global().canonical_species_name(name)
}

pub fn class_for_species(name: &str) -> Option<String> {
    global().class_for_species(name)
}

pub fn normalize_classification(raw: &str) -> String {
    global().normalize_classification(raw)
}

pub fn is_bird_class(raw: &str) -> bool {
    normalize_classification(raw) == "birds"
}

pub fn review_and_merge(table_path: &Path, merged_out: Option<&Path>) -> Result<ReviewReport> {
    let table = TaxonomyTable::load_from_file(table_path)?;

    let mut canonical_to_aliases: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (alias, canonical) in &table.alias_to_canonical {
        if alias == canonical {
            continue;
        }
        canonical_to_aliases
            .entry(canonical.clone())
            .or_default()
            .push(alias.clone());
    }

    for aliases in canonical_to_aliases.values_mut() {
        aliases.sort();
        aliases.dedup();
    }

    let mut species: Vec<MergedSpeciesAliases> = Vec::new();
    for (canonical, aliases) in canonical_to_aliases {
        species.push(MergedSpeciesAliases {
            class: table.canonical_class.get(&canonical).cloned(),
            canonical,
            aliases,
        });
    }

    species.sort_by(|a, b| a.canonical.cmp(&b.canonical));

    let class_aliases: BTreeMap<String, String> = table
        .class_aliases
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    let merged = MergedTaxonomyFile {
        species,
        class_aliases,
    };

    if let Some(path) = merged_out {
        // Serialize first so we catch any serialization error before touching disk.
        let text = toml::to_string_pretty(&merged)
            .context("Cannot serialize merged taxonomy table")?;

        // Create the output directory; warn and skip if it isn't writable
        // (e.g. the /data volume is read-only at entrypoint time).
        let can_write = if let Some(parent) = path.parent() {
            match std::fs::create_dir_all(parent) {
                Ok(()) => true,
                Err(e) => {
                    tracing::warn!(
                        "taxonomy: cannot create output directory {}: {e}; \
                         merged table will not be written",
                        parent.display()
                    );
                    false
                }
            }
        } else {
            true
        };

        if can_write {
            // Write atomically via a temp file so a crash mid-write
            // leaves the previous merged table intact.
            let tmp = path.with_extension("toml.tmp");
            match std::fs::write(&tmp, &text) {
                Ok(()) => {
                    if let Err(e) = std::fs::rename(&tmp, path) {
                        let _ = std::fs::remove_file(&tmp);
                        tracing::warn!(
                            "taxonomy: cannot rename temp file to {}: {e}; \
                             merged table will not be written",
                            path.display()
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "taxonomy: cannot write merged table to {}: {e}; \
                         merged table will not be written",
                        path.display()
                    );
                }
            }
        }
    }

    let report = ReviewReport {
        table_path: table_path.to_path_buf(),
        merged_path: merged_out.map(|p| p.to_path_buf()),
        species_count: merged.species.len(),
        alias_count: table
            .alias_to_canonical
            .iter()
            .filter(|(a, c)| *a != *c)
            .count(),
        class_alias_count: table.class_aliases.len(),
        class_override_count: table.canonical_class.len(),
    };

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_alias_to_canonical() {
        let t = TaxonomyTable {
            alias_to_canonical: HashMap::from([
                ("Psilorhinus morio".to_string(), "Cyanocorax morio".to_string()),
                ("Cyanocorax morio".to_string(), "Cyanocorax morio".to_string()),
            ]),
            canonical_class: HashMap::new(),
            class_aliases: HashMap::new(),
        };
        assert_eq!(
            t.canonical_species_name("Psilorhinus morio"),
            "Cyanocorax morio"
        );
    }

    #[test]
    fn normalizes_class_aliases() {
        let t = TaxonomyTable {
            alias_to_canonical: HashMap::new(),
            canonical_class: HashMap::new(),
            class_aliases: HashMap::from([
                ("aves".to_string(), "birds".to_string()),
                ("wildlife".to_string(), "wildlife".to_string()),
            ]),
        };
        assert_eq!(t.normalize_classification("AVES"), "birds");
        assert_eq!(t.normalize_classification("BIRDS"), "birds");
        assert_eq!(t.normalize_classification("WILDLIFE"), "wildlife");
    }
}