//! mDNS-SD (multicast DNS Service Discovery) for Gaia nodes.
//!
//! Each Gaia service (capture, processing, web) registers itself on the
//! local network via mDNS with a sequential instance name like
//! `capture-01`, `processing-02`, etc.
//!
//! The processing node uses discovery to locate capture nodes automatically,
//! removing the need for hard-coded URLs or DNS when running containers on
//! different hardware.

use std::collections::{BTreeSet, HashMap};
use std::net::IpAddr;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use tracing::{debug, info, warn};

/// How long to scan for existing peers before claiming an instance number.
const DISCOVERY_SCAN: Duration = Duration::from_secs(3);

// ── Service roles ────────────────────────────────────────────────────────────

/// The role a Gaia node plays on the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceRole {
    Capture,
    Processing,
    Web,
}

impl ServiceRole {
    /// mDNS service-type string including domain, e.g.
    /// `_gaia-aud-cap._tcp.local.`
    ///
    /// The service name (between `_` and `._tcp`) must be ≤ 15 bytes
    /// per RFC 6763 / DNS-SD.
    pub fn service_type(&self) -> &'static str {
        match self {
            Self::Capture => "_gaia-aud-cap._tcp.local.",
            Self::Processing => "_gaia-aud-proc._tcp.local.",
            Self::Web => "_gaia-aud-web._tcp.local.",
        }
    }

    /// Human-readable prefix used in instance names (e.g. `capture`).
    pub fn prefix(&self) -> &'static str {
        match self {
            Self::Capture => "capture",
            Self::Processing => "processing",
            Self::Web => "web",
        }
    }
}

// ── Peer ─────────────────────────────────────────────────────────────────────

/// A service discovered on the network.
#[derive(Debug, Clone)]
pub struct Peer {
    /// Instance name, e.g. `capture-01`.
    pub instance_name: String,
    /// All advertised IP addresses.
    pub addresses: Vec<IpAddr>,
    /// Listening port.
    pub port: u16,
}

impl Peer {
    /// Build an HTTP base URL for this peer, preferring IPv4.
    pub fn http_url(&self) -> Option<String> {
        let addr = self
            .addresses
            .iter()
            .find(|a| a.is_ipv4())
            .or_else(|| self.addresses.first())?;

        Some(match addr {
            IpAddr::V4(v4) => format!("http://{}:{}", v4, self.port),
            IpAddr::V6(v6) => format!("http://[{}]:{}", v6, self.port),
        })
    }

    /// All advertised addresses as non-loopback, preferring IPv4.
    pub fn non_loopback_addresses(&self) -> Vec<IpAddr> {
        let mut addrs: Vec<IpAddr> = self
            .addresses
            .iter()
            .filter(|a| !a.is_loopback())
            .copied()
            .collect();
        addrs.sort_by_key(|a| !a.is_ipv4()); // IPv4 first
        addrs
    }
}

// ── Discovery handle ─────────────────────────────────────────────────────────

/// Handle returned by [`register`].  Keeps the mDNS daemon alive and
/// provides methods to discover other peers.
pub struct DiscoveryHandle {
    daemon: ServiceDaemon,
    instance_name: String,
    fullname: String,
}

impl DiscoveryHandle {
    /// Our assigned instance name, e.g. `capture-01`.
    pub fn instance_name(&self) -> &str {
        &self.instance_name
    }

    /// Scan the network for peers of the given `role`.
    ///
    /// Blocks for up to `timeout` collecting advertisements, then returns
    /// the list of discovered peers (excluding ourselves).
    pub fn discover_peers(&self, role: ServiceRole, timeout: Duration) -> Vec<Peer> {
        let receiver = match self.daemon.browse(role.service_type()) {
            Ok(r) => r,
            Err(e) => {
                warn!("mDNS browse for {} failed: {e}", role.service_type());
                return vec![];
            }
        };

        debug!("mDNS: browsing for {} (timeout={}s)", role.service_type(), timeout.as_secs());
        // Collect by instance name so multiple ServiceResolved events
        // (one per interface / address family) are merged into a single peer.
        let mut peer_map: HashMap<String, Peer> = HashMap::new();
        let deadline = Instant::now() + timeout;

        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match receiver.recv_timeout(remaining) {
                Ok(ServiceEvent::ServiceResolved(info)) => {
                    let name = info.get_fullname().to_string();
                    // Don't include ourselves
                    if name == self.fullname {
                        debug!("mDNS: ignoring self ({})", name);
                        continue;
                    }
                    let addrs: Vec<IpAddr> =
                        info.get_addresses().iter().map(|a| a.to_ip_addr()).collect();
                    let port = info.get_port();
                    let instance = extract_instance_name(&name);

                    let peer = peer_map.entry(instance.clone()).or_insert_with(|| Peer {
                        instance_name: instance,
                        addresses: Vec::new(),
                        port,
                    });
                    // Merge addresses, avoiding duplicates.
                    for addr in addrs {
                        if !peer.addresses.contains(&addr) {
                            peer.addresses.push(addr);
                        }
                    }
                    debug!("mDNS: resolved {} – addrs now {:?}", name, peer.addresses);
                }
                Ok(event) => {
                    debug!("mDNS: event {:?}", format_event(&event));
                }
                Err(_) => break,
            }
        }

        let _ = self.daemon.stop_browse(role.service_type());
        if peer_map.is_empty() {
            debug!("mDNS: browse completed, no peers found for {}", role.service_type());
        }

        let peers: Vec<Peer> = peer_map.into_values().collect();
        for p in &peers {
            info!("mDNS: peer {} at {:?}:{}", p.instance_name, p.addresses, p.port);
        }
        peers
    }

    /// Unregister from mDNS and shut down the daemon.
    pub fn shutdown(self) {
        let _ = self.daemon.unregister(&self.fullname);
        let _ = self.daemon.shutdown();
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Register this node on the local network via mDNS.
///
/// The function scans for existing peers of the same role, picks the next
/// available sequential number, and registers an instance like
/// `capture-01` or `processing-03`.
pub fn register(role: ServiceRole, port: u16) -> Result<DiscoveryHandle> {
    let daemon = ServiceDaemon::new().context("Cannot start mDNS daemon")?;

    // ── scan for existing instances of the same role ────────────────
    let receiver = daemon
        .browse(role.service_type())
        .context("Cannot browse mDNS")?;

    let mut existing: BTreeSet<u32> = BTreeSet::new();
    let deadline = Instant::now() + DISCOVERY_SCAN;

    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        match receiver.recv_timeout(remaining) {
            Ok(ServiceEvent::ServiceResolved(info)) => {
                if let Some(n) =
                    parse_instance_number(info.get_fullname(), role.prefix())
                {
                    debug!("Found existing {} instance #{}", role.prefix(), n);
                    existing.insert(n);
                }
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }

    let _ = daemon.stop_browse(role.service_type());

    // ── pick next sequential number ─────────────────────────────────
    let our_number = next_available(&existing);
    let instance_name = format!("{}-{:02}", role.prefix(), our_number);
    let host = format!("{}.local.", instance_name);

    let service_info = ServiceInfo::new(
        role.service_type(),
        &instance_name,
        &host,
        "",   // filled automatically by enable_addr_auto()
        port,
        None, // no TXT properties
    )
    .context("Cannot create mDNS ServiceInfo")?
    .enable_addr_auto();

    let fullname = service_info.get_fullname().to_string();
    let registered_addrs = format!("{:?}", service_info.get_addresses());

    daemon
        .register(service_info)
        .context("Cannot register mDNS service")?;

    info!(
        "Registered on mDNS as '{}' (type={}, port={}, addrs={})",
        instance_name,
        role.service_type(),
        port,
        registered_addrs
    );

    Ok(DiscoveryHandle {
        daemon,
        instance_name,
        fullname,
    })
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Format a ServiceEvent for debug logging (without dumping the full struct).
fn format_event(event: &ServiceEvent) -> String {
    match event {
        ServiceEvent::ServiceFound(ty, name) => format!("Found({ty}, {name})"),
        ServiceEvent::ServiceResolved(info) => format!("Resolved({})", info.get_fullname()),
        ServiceEvent::ServiceRemoved(ty, name) => format!("Removed({ty}, {name})"),
        ServiceEvent::SearchStarted(ty) => format!("SearchStarted({ty})"),
        ServiceEvent::SearchStopped(ty) => format!("SearchStopped({ty})"),
        _ => format!("Other"),
    }
}

/// Extract the instance number from a fullname like
/// `capture-03._gaia-capture._tcp.local.`
fn parse_instance_number(fullname: &str, prefix: &str) -> Option<u32> {
    let instance = fullname.split('.').next()?;
    let suffix = instance.strip_prefix(prefix)?.strip_prefix('-')?;
    suffix.parse().ok()
}

/// Extract the short instance name from a fullname like
/// `capture-01._gaia-capture._tcp.local.`
fn extract_instance_name(fullname: &str) -> String {
    fullname
        .split('.')
        .next()
        .unwrap_or(fullname)
        .to_string()
}

/// Return the smallest positive integer not in `used`.
fn next_available(used: &BTreeSet<u32>) -> u32 {
    let mut n = 1;
    while used.contains(&n) {
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_instance_number() {
        assert_eq!(
            parse_instance_number("capture-01._gaia-capture._tcp.local.", "capture"),
            Some(1)
        );
        assert_eq!(
            parse_instance_number("processing-12._gaia-processing._tcp.local.", "processing"),
            Some(12)
        );
        assert_eq!(
            parse_instance_number("web-01._gaia-web._tcp.local.", "capture"),
            None
        );
        assert_eq!(
            parse_instance_number("garbage", "capture"),
            None
        );
    }

    #[test]
    fn test_next_available() {
        let empty = BTreeSet::new();
        assert_eq!(next_available(&empty), 1);

        let set: BTreeSet<u32> = [1, 2, 3].into();
        assert_eq!(next_available(&set), 4);

        let gap: BTreeSet<u32> = [1, 3].into();
        assert_eq!(next_available(&gap), 2);
    }

    #[test]
    fn test_extract_instance_name() {
        assert_eq!(
            extract_instance_name("capture-01._gaia-capture._tcp.local."),
            "capture-01"
        );
    }
}
