#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --job-name=transfer-archive
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 {fantasyid|az|wv} (ROC_TRANSFER_RAW_ROOT overrides shared data/raw)" >&2
    exit 2
fi
case "$1" in
    fantasyid)
        folder=fantasyid
        filename=FantasyID.tgz
        record=17063366
        expected_md5=5a6e608762f640db026ac122b4400a72
        ;;
    az)
        folder=idnet
        filename=AZ.zip
        record=13852757
        expected_md5=d51cdee3f3aaf4592039ba6623af7df3
        ;;
    wv)
        folder=idnet
        filename=WV.zip
        record=13854938
        expected_md5=a51cbc873c5cddd1b28b242f8bb96496
        ;;
    *) echo 'Unknown archive selector' >&2; exit 2 ;;
esac
: "${SLURM_JOB_ID:?Allocated Slurm job required}"
: "${SLURMD_NODENAME:?Allocated compute node required}"
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
case "${SLURMD_NODENAME,,}" in
    *login*|*head*|*frontend*|*lgn*) echo 'Downloads are forbidden on login nodes' >&2; exit 2 ;;
esac
[[ "${SLURM_JOB_PARTITION:-}" == cpu ]] || { echo 'Allocated CPU partition required' >&2; exit 2; }
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || { echo 'Invalid Slurm job ID' >&2; exit 2; }
command -v curl >/dev/null || { echo 'Required download tool unavailable: curl' >&2; exit 2; }
command -v md5sum >/dev/null || { echo 'Required checksum tool unavailable: md5sum' >&2; exit 2; }
cd "$SLURM_SUBMIT_DIR"
raw_root="${ROC_TRANSFER_RAW_ROOT:-/shared/b00090279/refer-or-clear/data/raw}"
[[ "$raw_root" == /* && ! -L "$raw_root" && ! -L "$raw_root/$folder" ]] || {
    echo 'Absolute nonsymlink raw destination required' >&2; exit 2;
}
mkdir -p -- "$raw_root/$folder"
cd "$raw_root/$folder"
part="$filename.part"
lock="$filename.lock"
mkdir -- "$lock" || { echo 'Archive already locked; inspect owner before removing a stale lock' >&2; exit 2; }
trap 'rmdir -- "$lock" || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
[[ ! -L "$filename" && ! -L "$part" ]] || { echo 'Archive symlinks are forbidden' >&2; exit 2; }
verify_md5() {
    printf '%s  %s\n' "$expected_md5" "$1" | md5sum --check --status
}
if [[ -e "$filename" ]]; then
    [[ -f "$filename" ]] && verify_md5 "$filename" || {
        echo 'Existing archive is not checksum verified; refusing to overwrite' >&2; exit 2;
    }
    echo "Verified existing archive: $PWD/$filename"
    exit 0
fi
[[ ! -e "$part" || -f "$part" ]] || { echo 'Partial archive must be a regular file' >&2; exit 2; }
url="https://zenodo.org/records/$record/files/$filename?download=1"
curl --disable --request GET --fail --location --retry 3 --retry-delay 5 --connect-timeout 30 \
    --continue-at - --output "$part" "$url"
verify_md5 "$part" || { echo 'Downloaded archive MD5 mismatch; partial retained, final not created' >&2; exit 2; }
mv -n -- "$part" "$filename"
[[ ! -e "$part" ]] || { echo 'Final appeared during download; refusing to overwrite' >&2; exit 2; }
echo "Verified downloaded archive: $PWD/$filename"
