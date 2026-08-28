# Deploy and rollback

Build by immutable digest. Deploy only the inactive slot, require `/health/ready`,
then switch the gateway by a graceful reload and drain the old slot. Webhooks must
persist before returning 202. Workers stop new claims on SIGTERM and finish current
leases. Never run a migration here. Rollback switches only transport by digest.
