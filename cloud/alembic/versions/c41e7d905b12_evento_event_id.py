"""evento.event_id: ingestión idempotente

`POST /devices/events` guardaba dos veces un lote si la nube lo confirmaba pero la respuesta no
llegaba a la controladora (ver "Ingestión idempotente" en `cloud/README.md`). Cada evento trae
ahora un identificador generado en la controladora (`MonitoringEvent.event_id`) y una cámara no
puede tener dos filas con el mismo.

- EVENTO: + `event_id` (null), único junto con `camara_id`.

Datos existentes: quedan con `event_id` en NULL, que no cuenta como repetido. No hace falta
completarlos: la controladora no vuelve a enviar lo que ya marcó como sincronizado.

Usa `batch_alter_table` para que funcione igual en SQLite (desarrollo) y Postgres (RDS).

Revision ID: c41e7d905b12
Revises: a3f9c2d17b40
Create Date: 2026-10-13 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c41e7d905b12'
down_revision: Union[str, Sequence[str], None] = 'a3f9c2d17b40'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('eventos') as batch_op:
        batch_op.add_column(sa.Column('event_id', sa.String(length=64), nullable=True))
        batch_op.create_unique_constraint('uq_eventos_camara_event_id', ['camara_id', 'event_id'])


def downgrade() -> None:
    with op.batch_alter_table('eventos') as batch_op:
        batch_op.drop_constraint('uq_eventos_camara_event_id', type_='unique')
        batch_op.drop_column('event_id')
