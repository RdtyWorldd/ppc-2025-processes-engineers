if (rank == 0) {
  std::vector<double> &input_a = std::get<1>(GetInput());
  std::vector<double> &input_b = std::get<2>(GetInput());

  int step = n / mpi_size;
  std::vector<int> start_i(mpi_size);
  std::vector<int> end_i(mpi_size);
  for (int i = 0; i < mpi_size - 1; i++) {
    start_i[i] = i * step;
    end_i[i] = (i + 1) * step;
  }
  start_i[(mpi_size - 1)] = (mpi_size - 1) * step;
  end_i[(mpi_size - 1)] = n;

  // рассылка
  MPI_Request requests[(mpi_size - 1) * 3];
  MPI_Status statuses[(mpi_size - 1) * 3];
  for (int i = 1; i < mpi_size; i++) {
    int send_data_size = end_i[i] - start_i[i];
    MPI_ISend(send_data_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, requests + (i * 3));
    MPI_ISend(input_a.data() + start_i[i], send_data_size, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, requests + ((i * 3) + 1));
    MPI_ISend(input_b.data() + start_i[i], send_data_size, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, requests + ((i * 3) + 2));
  }

  MPI_Waitall((mpi_size - 1) * 3, requests, statuses);
} else {
  int recv_data_size = 0;
  MPI_Recv(&recv_data_size, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  a.resize(recv_data_size);
  b.resize(recv_data_size);

  MPI_Recv(a.data(), recv_data_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(b.data(), recv_data_size, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
