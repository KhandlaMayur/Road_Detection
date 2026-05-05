"""
Run this script ONCE to generate cert.pem and key.pem for HTTPS.
Usage: python generate_cert.py
"""
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import datetime
import ipaddress
import socket

hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
print(f"Generating certificate for IP: {local_ip}, hostname: {hostname}")

key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
    backend=default_backend()
)

now = datetime.datetime.now(datetime.timezone.utc)

subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, u"IN"),
    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"Gujarat"),
    x509.NameAttribute(NameOID.LOCALITY_NAME, u"Surat"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"RoadGuard AI"),
    x509.NameAttribute(NameOID.COMMON_NAME, u"roadguard.local"),
])

cert = (
    x509.CertificateBuilder()
    .subject_name(subject)
    .issuer_name(issuer)
    .public_key(key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(now)
    .not_valid_after(now + datetime.timedelta(days=3650))
    .add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName(u"localhost"),
            x509.DNSName(hostname),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            x509.IPAddress(ipaddress.IPv4Address(local_ip)),
        ]),
        critical=False,
    )
    .sign(key, hashes.SHA256(), default_backend())
)

with open("key.pem", "wb") as f:
    f.write(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()
    ))

with open("cert.pem", "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))

print("cert.pem and key.pem created!")
print(f"Open on your PC:     https://localhost:5000")
print(f"Open on your phone:  https://{local_ip}:5000")
