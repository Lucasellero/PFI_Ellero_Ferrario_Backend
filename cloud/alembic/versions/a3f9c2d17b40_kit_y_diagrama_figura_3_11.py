"""kit y resto del diagrama de la Figura 3.11

Alinea el esquema con el diagrama de base de datos en la nube del informe (detalle en
`docs/vinculacion-kit.md` del repo Frontend, sección 4):

- KIT nueva; `device_api_key_hash` pasa de HOGAR a KIT.
- HOGAR: + `direccion` (null), + `fecha_alta`.
- CAMARA: + `kit_id`, `hogar_id` pasa a null, `local_id` -> `identificador_local` (único por kit),
  `ubicacion` obligatoria ("" = sin cargar todavía).
- RESIDENTE: + `telefono`. FAMILIAR: + `push_token`. EVENTO: + `resuelta_at`, `resuelta_por`.
- LINEA_BASE_COMPORTAMIENTO nueva.
- HOGAR_FAMILIAR.rol: "responsable" (default viejo) -> "responsable_principal".

Datos existentes: cada hogar recibe un kit `LEGACY-NNNNNN` con SU MISMA API key (la Pi sigue
autenticando sin tocar `cameras.yaml`) y sus cámaras pasan a ese kit. Ese kit ya figura vinculado;
su código de vinculación es aleatorio y no se muestra (no hace falta: ya está vinculado).

Usa `batch_alter_table` para que funcione igual en SQLite (desarrollo) y Postgres (RDS).

Revision ID: a3f9c2d17b40
Revises: ebcb16f6fd06
Create Date: 2026-10-06 21:00:00.000000

"""
import hashlib
import secrets
from datetime import datetime, timezone
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a3f9c2d17b40'
down_revision: Union[str, Sequence[str], None] = 'ebcb16f6fd06'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _hash_aleatorio() -> str:
    return hashlib.sha256(secrets.token_bytes(32)).hexdigest()


def upgrade() -> None:
    conn = op.get_bind()
    ahora = datetime.now(timezone.utc)

    # --- KIT ---------------------------------------------------------------------------------
    op.create_table('kits',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('numero_serie', sa.String(length=20), nullable=False),
        sa.Column('cantidad_camaras', sa.Integer(), nullable=False),
        sa.Column('codigo_vinculacion_hash', sa.String(length=64), nullable=False),
        sa.Column('device_api_key_hash', sa.String(length=64), nullable=False),
        sa.Column('hogar_id', sa.Integer(), nullable=True),
        sa.Column('vinculado_por', sa.Integer(), nullable=True),
        sa.Column('vinculado_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('ultima_conexion_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('fabricado_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['hogar_id'], ['hogares.id']),
        sa.ForeignKeyConstraint(['vinculado_por'], ['familiares.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('numero_serie'),
    )
    op.create_index(op.f('ix_kits_codigo_vinculacion_hash'), 'kits', ['codigo_vinculacion_hash'], unique=True)
    op.create_index(op.f('ix_kits_device_api_key_hash'), 'kits', ['device_api_key_hash'], unique=True)
    op.create_index(op.f('ix_kits_hogar_id'), 'kits', ['hogar_id'], unique=False)

    kits = sa.table('kits', *(sa.column(c) for c in (
        'numero_serie', 'cantidad_camaras', 'codigo_vinculacion_hash', 'device_api_key_hash',
        'hogar_id', 'vinculado_at', 'fabricado_at')))
    hogares_existentes = conn.execute(sa.text(
        "SELECT h.id, h.device_api_key_hash, "
        "(SELECT COUNT(*) FROM camaras c WHERE c.hogar_id = h.id) FROM hogares h"
    )).fetchall()
    for hogar_id, api_key_hash, n_camaras in hogares_existentes:
        conn.execute(kits.insert().values(
            numero_serie=f"LEGACY-{hogar_id:06d}", cantidad_camaras=max(1, n_camaras),
            codigo_vinculacion_hash=_hash_aleatorio(), device_api_key_hash=api_key_hash,
            hogar_id=hogar_id, vinculado_at=ahora, fabricado_at=ahora,
        ))

    # --- HOGAR -------------------------------------------------------------------------------
    op.drop_index(op.f('ix_hogares_device_api_key_hash'), table_name='hogares')
    with op.batch_alter_table('hogares') as b:
        b.add_column(sa.Column('direccion', sa.String(length=300), nullable=True))
        b.add_column(sa.Column('fecha_alta', sa.DateTime(timezone=True), nullable=True))
    conn.execute(sa.text("UPDATE hogares SET fecha_alta = :ahora"), {"ahora": ahora})
    with op.batch_alter_table('hogares') as b:
        b.alter_column('fecha_alta', existing_type=sa.DateTime(timezone=True), nullable=False)
        b.drop_column('device_api_key_hash')

    # --- CAMARA ------------------------------------------------------------------------------
    with op.batch_alter_table('camaras') as b:
        b.add_column(sa.Column('kit_id', sa.Integer(), nullable=True))
    conn.execute(sa.text(
        "UPDATE camaras SET kit_id = (SELECT k.id FROM kits k WHERE k.hogar_id = camaras.hogar_id)"
    ))
    conn.execute(sa.text("UPDATE camaras SET ubicacion = '' WHERE ubicacion IS NULL"))
    with op.batch_alter_table('camaras') as b:
        b.alter_column('kit_id', existing_type=sa.Integer(), nullable=False)
        b.alter_column('hogar_id', existing_type=sa.Integer(), nullable=True)
        b.alter_column('ubicacion', existing_type=sa.String(length=200), nullable=False, server_default='')
        b.alter_column('local_id', new_column_name='identificador_local',
                       existing_type=sa.String(length=100), existing_nullable=False)
        b.create_foreign_key('fk_camaras_kit_id_kits', 'kits', ['kit_id'], ['id'])
        b.create_index(op.f('ix_camaras_kit_id'), ['kit_id'], unique=False)
    # en un batch aparte: en SQLite, el UNIQUE sobre una columna renombrada en el mismo batch se pierde
    with op.batch_alter_table('camaras') as b:
        b.create_unique_constraint('uq_camaras_kit_identificador_local', ['kit_id', 'identificador_local'])

    # --- RESIDENTE / FAMILIAR / EVENTO -------------------------------------------------------
    with op.batch_alter_table('residentes') as b:
        b.add_column(sa.Column('telefono', sa.String(length=20), nullable=True))
    with op.batch_alter_table('familiares') as b:
        b.add_column(sa.Column('push_token', sa.String(length=300), nullable=True))
    with op.batch_alter_table('eventos') as b:
        b.add_column(sa.Column('resuelta_at', sa.DateTime(timezone=True), nullable=True))
        b.add_column(sa.Column('resuelta_por', sa.Integer(), nullable=True))
        b.create_foreign_key('fk_eventos_resuelta_por_familiares', 'familiares', ['resuelta_por'], ['id'])

    # --- HOGAR_FAMILIAR: valores de rol definidos en el informe -------------------------------
    conn.execute(sa.text("UPDATE hogar_familiar SET rol = 'responsable_principal' WHERE rol = 'responsable'"))

    # --- LINEA_BASE_COMPORTAMIENTO -------------------------------------------------------------
    op.create_table('lineas_base_comportamiento',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('residente_id', sa.Integer(), nullable=False),
        sa.Column('parametros_cluster', sa.JSON(), nullable=False),
        sa.Column('actualizado_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['residente_id'], ['residentes.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('residente_id'),
    )


def downgrade() -> None:
    """Vuelve al esquema anterior. Pierde: kits sin vincular y sus cámaras (en el esquema viejo
    una cámara necesita hogar), kits viejos de un hogar (queda la API key del kit activo),
    teléfonos, push tokens, resoluciones de alertas y líneas base."""
    conn = op.get_bind()

    op.drop_table('lineas_base_comportamiento')
    conn.execute(sa.text("UPDATE hogar_familiar SET rol = 'responsable' WHERE rol = 'responsable_principal'"))

    with op.batch_alter_table('eventos') as b:
        b.drop_constraint('fk_eventos_resuelta_por_familiares', type_='foreignkey')
        b.drop_column('resuelta_por')
        b.drop_column('resuelta_at')
    with op.batch_alter_table('familiares') as b:
        b.drop_column('push_token')
    with op.batch_alter_table('residentes') as b:
        b.drop_column('telefono')

    conn.execute(sa.text(
        "UPDATE eventos SET camara_id = NULL WHERE camara_id IN (SELECT id FROM camaras WHERE hogar_id IS NULL)"
    ))
    conn.execute(sa.text("DELETE FROM camaras WHERE hogar_id IS NULL"))
    with op.batch_alter_table('camaras') as b:
        b.drop_index(op.f('ix_camaras_kit_id'))
        b.drop_constraint('uq_camaras_kit_identificador_local', type_='unique')
        b.drop_constraint('fk_camaras_kit_id_kits', type_='foreignkey')
        b.alter_column('identificador_local', new_column_name='local_id',
                       existing_type=sa.String(length=100), existing_nullable=False)
        b.alter_column('hogar_id', existing_type=sa.Integer(), nullable=False)
        b.alter_column('ubicacion', existing_type=sa.String(length=200), nullable=True, server_default=None)
        b.drop_column('kit_id')
    conn.execute(sa.text("UPDATE camaras SET ubicacion = NULL WHERE ubicacion = ''"))

    with op.batch_alter_table('hogares') as b:
        b.add_column(sa.Column('device_api_key_hash', sa.String(length=64), nullable=True))
    for (hogar_id,) in conn.execute(sa.text("SELECT id FROM hogares")).fetchall():
        fila = conn.execute(sa.text(
            "SELECT device_api_key_hash FROM kits WHERE hogar_id = :h "
            "ORDER BY vinculado_at DESC LIMIT 1"), {"h": hogar_id}).fetchone()
        conn.execute(sa.text("UPDATE hogares SET device_api_key_hash = :k WHERE id = :h"),
                     {"k": fila[0] if fila else _hash_aleatorio(), "h": hogar_id})
    with op.batch_alter_table('hogares') as b:
        b.alter_column('device_api_key_hash', existing_type=sa.String(length=64), nullable=False)
        b.drop_column('fecha_alta')
        b.drop_column('direccion')
    op.create_index(op.f('ix_hogares_device_api_key_hash'), 'hogares', ['device_api_key_hash'], unique=True)

    op.drop_index(op.f('ix_kits_hogar_id'), table_name='kits')
    op.drop_index(op.f('ix_kits_device_api_key_hash'), table_name='kits')
    op.drop_index(op.f('ix_kits_codigo_vinculacion_hash'), table_name='kits')
    op.drop_table('kits')
